import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
             save_intermediate=False, tokenizer=None, output_file="generation_process.txt"):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        save_intermediate: Whether to save intermediate results at each timestep to a file.
        tokenizer: Tokenizer for decoding intermediate results.
        output_file: Name of the file to save intermediate results.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):

        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            
            # Save intermediate result to file if requested

            if save_intermediate and tokenizer is not None:
                ids = x[0, prompt.shape[1]:].detach().tolist()  # 生成部分 token ids（在GPU也行，tolist会拷到CPU）
                parts = []
                buf = []

                for tid in ids:
                    if tid == mask_id:
                        # 先把之前累计的“已确定段”解码
                        if buf:
                            txt = tokenizer.decode(buf, skip_special_tokens=False)
                            # 防止控制字符污染日志（可选但建议）
                            txt = txt.replace("\x00", "")
                            parts.append(txt)
                            buf = []
                        parts.append("<MASK>")
                    else:
                        buf.append(tid)

                # 收尾
                if buf:
                    txt = tokenizer.decode(buf, skip_special_tokens=False)
                    txt = txt.replace("\x00", "")
                    parts.append(txt)

                intermediate_text = "".join(parts)

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"Block {num_block+1} Step {i+1}/{steps} transferred={int(transfer_index[0].sum().item())}\n")
                    f.write(intermediate_text + "\n")
                    f.write("-" * 50 + "\n")

    return x


def main():
    device = 'cuda'

    model_path = '/data/labshare/Param/llada/'
    #model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    #tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    #"Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
    #"Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"
    prompts = '''
    [
        {
            "role": "system",
            "content": "You are a helpful assistant to make travel plans for Bob.\n\nEXTERNAL RESOURCES:\n1. A database containing information about train tickets, attractions, and city transportation.\n2. A python notebook to execute python code for numerical operations and planning. \n\nTASK DESCRIPTION\nYou need to make a travel plan based on the given requirements, taking into account transportation between cities and daily schedules.\nThe final travel plan may include or be part of the following:\n1.go_to_place(origin:str,destination:str,departure_time,arrival_time): go to destination from origin. The origin and destination should be the name of a hotel or a spot instead of a city.\n2.visit(place:str,begin_time,end_time): visit somewhere from begin_time to end_time. The time should be expressed as \"%Y-%m-%d %H:%M\", e.g. 2023-07-02 16:00. You have to go somewhere before you can visit it.\n3.go_to_city(origin_city:str,destination_city:str,departure_time,arrival_time,ticket_number): go to destination city from origin city, using the ticket with the ticket_number(you can know the ticket number from the database).\n4.stay_in(city:str,begin_time,end_time): stay in somewhere from begin_time to end_time. The time should be expressed as \"%Y-%m-%d %H:%M\". Only when Bob is in some city can he visit it.\ne.g. \n<plan>go_to_place(\"Beijing Railway Hotel\",\"The Great Wall\",\"2023-07-02 7:00\",\"2023-07-02 8:05\")</plan>, <plan>visit(\"The Great Wall\",\"2023-07-02 8:05\",\"2023-07-05 17:00\")</plan>,<plan>go_to_city(\"Shanghai\",\"Beijing\",\"2023-07-02 16:00\",\"2023-07-02 22:30\",\"D1111\")</plan>, <plan>stay_in(\"Beijing\",\"2023-07-02 22:30\",\"2023-07-05 8:00\")</plan>\nYour ultimate goal is to give these plans, there is no need to do anything extra.\n\n--- Your Workflow ---\n1. You will first be given a task.\n2. You should understand the task and devise a plan to complete the task. This plan will contain a series of subtasks that need to be completed.\n\nPLAN AND SUBTASK:\nIf the task cannot be easily solved directly or requires the use of external resources, please assign it to another agent to complete (such as \"find the cheapest train from Beijing to Shanghai in 2023-7-1\"), otherwise you can complete it yourself. You may need to wait for results from other agents before proceeding to the next step of the task. If you need help from other agents, please clearly describe the task objectives, background, and precautions of the subtask. \nA subtask-structure has the following json component and surrounded with <subtask></subtask> as follows:\n<subtask>{\n\"subtask_name\": string, name of the subtask\n\"goal\": string, the main purpose of the subtask, and what will you do to reach this goal?\n\"criticism\": string, what potential problems may the current subtask and goal have?\n\"milestones\": list[string]. what milestones should be achieved to ensure the subtask is done? Make it detailed and specific.\n\"result_format\": optional, what the result should be.}</subtask>\n\n"
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Task Requirements: Bob is in Shanghai and going to travel in several cities, please make a ticket purchase plan and travel sequence for him.The demands are as follows:\n1. visit ['Beijing']. The order doesn't matter and he needs to return to Shanghai finally.\n2. He is free to travel from 2023.7.1 to 2023.7.20. The budget for transportation is 1000.0 CNY.\n3. Play at least 3 days in Beijing.\n4. If you arrive in a city before 12:00 noon, that day can be counted as a day of play. If it's past 12 o'clock, it doesn't count as a day.\n5. On the basis of completing the above conditions (especially the budget), spend as little time as possible.\n"
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Based on the requirements, we can know that Bob need to go to Beijing from Shanghai, stay in Beijing for 3 days and then go to Shanghai from Beijing.\nGiven the task, the first step is to find available train tickets that fit Bob's schedule and budget.\n<subtask>\n{\n\"subtask_name\": \"find_available_train_tickets\",\n\"goal\": \"Find train tickets from Shanghai to Beijing and back to Shanghai that fit within the travel dates, budget, and allow for at least 3 full days of play in Beijing. If the arrival is before 12:00 noon, it counts as a day of play.\",\n\"criticism\": \"Must ensure that the total cost of the round trip tickets does not exceed the budget of 1000.0 CNY and that the timings allow for at least 3 full days in Beijing for visit. So you need to allow time between train rides(arrival in a city and departure from the city). For each ticket, you must give me the ticket number, origin, destination, departure time, arrival time and the price.\",\n\"milestones\": [\"Identify a suitable train from Shanghai to Beijing.\", \"Identify a return train from Beijing to Shanghai ensuring at least 3 days in Beijing before departing.\", \"Ensure the total cost of both tickets is within the budget of 1000.0 CNY.\"]\n}\n</subtask>\nThen we can get the final plan consists of go_to_city and stay_in.\n<subtask>\n{\n\"subtask_name\": \"get the final plan\",\n\"goal\": \"Formulate a travel plan for Bob's trip from Shanghai to Beijing and back, ensuring it fits within his budget and time constraints, including at least 3 full days in Beijing.\",\n\"criticism\": \"The plan must be concise, focusing on efficient travel and stay arrangements while adhering to the budget and time constraints.\",\n\"milestones\": [\"Include suitable train journeys within the budget.\",\"Plan at least 3 full days in Beijing.\",\"Ensure the overall plan fits within the specified dates and budget.\"],\n\"result_format\": \"A schedule consisting with multiple <plan>go_to_place(...)</plan> and <plan>visit(...)</plan>.    1.go_to_place(origin:str,destination:str,departure_time,arrival_time): go to destination from origin.     2.visit(place:str,begin_time,end_time): visit somewhere from begin_time to end_time. The time should be expressed as %Y-%m-%d %H:%M, e.g. 2023-07-02 16:00.\"\n}\n</subtask>\n"
        },
        {
            "role": "user",
            "content": "Task Requirements:\nBob is in Shanghai and going to travel in several cities, please make a ticket purchase plan and travel sequence for him.The demands are as follows:\n1. visit ['Hangzhou']. The order doesn't matter and he needs to return to Shanghai finally.\n2. 5 days (2023-07-03 00:00 ~ 2023-07-08 00:00) for this trip.\n3. Play at least 1 day in Hangzhou.\n4. Stay in any city for a minimum of 24 hours to count as one day.\n5. On the basis of completing the above conditions (especially the time limit), spend as little money as possible.\nCome up with an abstract plan to perform this task in a couple of steps. Give me the subtasks between <subtask> and </subtask>."
        }
    ]
    '''

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompts}]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=1024, block_length=256, temperature=0., cfg_scale=0., remasking='low_confidence',save_intermediate=True, tokenizer=tokenizer, output_file="denoise_log_128_128_128.txt")
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()

