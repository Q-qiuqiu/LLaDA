import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    """
    Gumbel-max sampling helper. Uses float64 noise as in some MDM work.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def find_subseq_1d(haystack, needle):
    """Return first start index of needle in haystack (both list[int]), else -1."""
    n = len(needle)
    if n == 0 or len(haystack) < n:
        return -1
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def safe_topk(conf_row, k: int):
    """conf_row: (L,) tensor. Return (values, indices) or None if no finite entries."""
    finite = torch.isfinite(conf_row)
    avail = int(finite.sum().item())
    if avail <= 0:
        return None
    k = min(int(k), avail)
    if k <= 0:
        return None
    return torch.topk(conf_row, k=k)


def transfer_k(remaining: int, steps_left: int, min_k: int = 1, max_k: int | None = None) -> int:
    """
    Choose how many tokens to transfer this step so that we can finish by the end.
    Uses ceil(remaining / steps_left).
    """
    if steps_left <= 0:
        return remaining
    k = (remaining + steps_left - 1) // steps_left
    k = max(min_k, k)
    if max_k is not None:
        k = min(k, max_k)
    return k


@torch.no_grad()
def generate_think_then_tool_but_decode_tool_first(
    model,
    prompt_ids,
    attention_mask=None,
    steps=128,
    think_budget=128,
    tool_budget=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    tokenizer=None,
    # 建议打开，避免 tool 槽位里被 eot/eos 填坏
    suppress_eos_eot=True,
    eos_id=126081,   # 你原代码里用的
    eot_id=126348,   # 你原代码里用的
    save_intermediate=False,
    output_file="denoise_log.txt",
):
    """
    最终输出结构： <THINK> ... </THINK> <TOOL_CALL> ... </TOOL_CALL>
    但采样顺序：先把 TOOL_CALL 槽位填满，再填 THINK 槽位。
    Tool 槽位内部仍然是默认的 LLaDA 风格：每步选高置信 token 先 transfer（不强制顺序）。
    """

    assert tokenizer is not None, "需要 tokenizer 才能写入/解码 marker"

    device = model.device
    B, prompt_len = prompt_ids.shape

    # 固定 marker（不参与生成）
    think_open  = tokenizer.encode("<THINK>\n", add_special_tokens=False)
    think_close = tokenizer.encode("\n</THINK>\n", add_special_tokens=False)
    tool_open   = tokenizer.encode("<TOOL_CALL>\n", add_special_tokens=False)
    tool_close  = tokenizer.encode("\n</TOOL_CALL>\n", add_special_tokens=False)

    total_gen_len = (
        len(think_open) + think_budget + len(think_close) +
        len(tool_open) + tool_budget + len(tool_close)
    )

    # 初始化：prompt + 全 mask 生成区
    x = torch.full((B, prompt_len + total_gen_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids.clone()

    # attention_mask 扩展
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask.to(device),
             torch.ones((B, total_gen_len), dtype=attention_mask.dtype, device=device)],
            dim=-1
        )

    # 把 marker 写进 x，并记录两个槽位范围
    pos = prompt_len

    def write_ids(ids):
        nonlocal pos
        t = torch.tensor(ids, dtype=torch.long, device=device)
        x[:, pos:pos + len(ids)] = t
        pos += len(ids)

    write_ids(think_open)
    think_slot_s = pos
    think_slot_e = think_slot_s + think_budget
    pos = think_slot_e
    write_ids(think_close)

    write_ids(tool_open)
    tool_slot_s = pos
    tool_slot_e = tool_slot_s + tool_budget
    pos = tool_slot_e
    write_ids(tool_close)

    gen_end = pos  # 生成区实际结束（绝对索引）

    # 阶段控制：先 tool 后 think
    in_tool_stage = True
    tool_steps_used = 0
    think_steps_used = 0

    # 给 tool/think 分配“最多步数”（tool 填满会提前切换并把剩余步数给 think）
    tool_steps_max = steps // 2
    think_steps_max = steps - tool_steps_max

    # 预先保存一次文件头（可选）
    if save_intermediate:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"steps={steps}, think_budget={think_budget}, tool_budget={tool_budget}\n")
            f.write("=" * 60 + "\n")

    for t in range(steps):
        mask_index = (x == mask_id)

        # CFG（可选）
        if cfg_scale > 0.0:
            prompt_index = (x != mask_id)  # prompt + marker 都算已知
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            if attention_mask is not None:
                attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
            else:
                attention_mask_ = None
            logits = model(x_, attention_mask=attention_mask_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x, attention_mask=attention_mask).logits

        if suppress_eos_eot:
            logits[:, :, eos_id] = -torch.inf
            logits[:, :, eot_id] = -torch.inf

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        # 置信度（用于 low_confidence remasking）
        if remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        elif remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=device)
        else:
            raise NotImplementedError(remasking)

        # 只允许在当前槽位 transfer（其余位置 confidence=-inf）
        eligible = torch.zeros_like(x0_p, dtype=torch.bool, device=device)
        if in_tool_stage:
            eligible[:, tool_slot_s:tool_slot_e] = True
        else:
            eligible[:, think_slot_s:think_slot_e] = True

        # 只在 mask 位置上考虑
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index & eligible, x0_p, -float("inf"))

        # 计算本步要 transfer 的数量：确保在分配到的剩余步数内能填满
        if in_tool_stage:
            remaining = int((x[:, tool_slot_s:tool_slot_e] == mask_id).sum().item())
            steps_left = max(1, tool_steps_max - tool_steps_used)
            k_this_step = transfer_k(remaining, steps_left, min_k=1)
        else:
            remaining = int((x[:, think_slot_s:think_slot_e] == mask_id).sum().item())
            steps_left = max(1, think_steps_max - think_steps_used)
            k_this_step = transfer_k(remaining, steps_left, min_k=1)

        transfer_index = torch.zeros_like(x, dtype=torch.bool, device=device)
        for j in range(B):
            out = safe_topk(confidence[j], k_this_step)
            if out is None:
                continue
            _, select_index = out
            transfer_index[j, select_index] = True

        x[transfer_index] = x0[transfer_index]

        # 记录中间态（可选，很慢）
        if save_intermediate:
            ids = x[0, prompt_len:gen_end].detach().tolist()
            parts = []
            buf = []
            for tid in ids:
                if tid == mask_id:
                    if buf:
                        txt = tokenizer.decode(buf, skip_special_tokens=False).replace("\x00", "")
                        parts.append(txt)
                        buf = []
                    parts.append("<MASK>")
                else:
                    buf.append(tid)
            if buf:
                txt = tokenizer.decode(buf, skip_special_tokens=False).replace("\x00", "")
                parts.append(txt)
            intermediate_text = "".join(parts)

            with open(output_file, "a", encoding="utf-8") as f:
                stage = "TOOL" if in_tool_stage else "THINK"
                f.write(f"Step {t+1}/{steps} stage={stage} k={k_this_step} remaining={remaining}\n")
                f.write(intermediate_text + "\n")
                f.write("-" * 60 + "\n")

        # 阶段切换逻辑
        if in_tool_stage:
            tool_steps_used += 1
            tool_remaining = int((x[:, tool_slot_s:tool_slot_e] == mask_id).sum().item())
            # tool 填满 或 用完 tool_steps_max：切到 think
            if tool_remaining == 0 or tool_steps_used >= tool_steps_max:
                in_tool_stage = False
        else:
            think_steps_used += 1
            # think 填满可提前结束（可选）
            think_remaining = int((x[:, think_slot_s:think_slot_e] == mask_id).sum().item())
            if think_remaining == 0:
                break

    return x, (think_slot_s, think_slot_e), (tool_slot_s, tool_slot_e), gen_end


def main():
    device = "cuda"

    model_path = "/home/yzx/models_weight/LLaDA/"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.pad_token_id != 126336

    prompts = """
You are an intelligent assistant designed to help users accomplish tasks using a set of tools.

Output format (MUST):
<THINK>
...your reasoning...
</THINK>
<TOOL_CALL>
{ "server_name": "...", "tool_name": "...", "params": {...} }
</TOOL_CALL>

Query: I'm planning a trip to the Forbidden City in Beijing, show me where to park and what's interesting to do within 1km distance?

Available MCP tools:

- suggest_meeting_point (server: osm-mcp-server): 
Find the optimal meeting place for multiple people coming from different locations.

This tool calculates a central meeting point based on the locations of multiple individuals,
then recommends suitable venues near that central point. Ideal for planning social gatherings,
business meetings, or any situation where multiple people need to converge from different
starting points.

Args:
    locations: List of dictionaries, each containing the latitude and longitude of a person's location
            Example: [{"latitude": 37.7749, "longitude": -122.4194}, {"latitude": 37.3352, "longitude": -121.8811}]
    venue_type: Type of venue to suggest as a meeting point. Options include:
            "cafe", "restaurant", "bar", "library", "park", etc.
    
Returns:
    Meeting point recommendations including:
    - Calculated center point coordinates
    - List of suggested venues with names and details
    - Total number of matching venues in the area
. Input: {"locations": {"items": {"additionalProperties": {"type": "number"}, "type": "object"}, "title": "Locations", "type": "array"}, "venue_type": {"default": "cafe", "title": "Venue Type", "type": "string"}}
- explore_area (server: osm-mcp-server): 
Generate a comprehensive profile of an area including all amenities and features.

This powerful analysis tool creates a detailed overview of a neighborhood or area by
identifying and categorizing all geographic features, amenities, and points of interest.
Results are organized by category for easy analysis. Excellent for neighborhood research,
area comparisons, and location-based decision making.

Args:
    latitude: Center point latitude (decimal degrees)
    longitude: Center point longitude (decimal degrees)
    radius: Search radius in meters (defaults to 500m)
    
Returns:
    In-depth area profile including:
    - Address and location context
    - Total feature count
    - Features organized by category and subcategory
    - Each feature includes name, coordinates, and detailed metadata
. Input: {"latitude": {"title": "Latitude", "type": "number"}, "longitude": {"title": "Longitude", "type": "number"}, "radius": {"default": 500, "title": "Radius", "type": "number"}}
- find_ev_charging_stations (server: osm-mcp-server): 
Locate electric vehicle charging stations near a specific location.

This specialized search tool identifies EV charging infrastructure within a specified
distance from a location. Results can be filtered by connector type (Tesla, CCS, CHAdeMO, etc.)
and minimum power delivery. Essential for EV owners planning trips or evaluating potential
charging stops.

Args:
    latitude: Center point latitude (decimal degrees)
    longitude: Center point longitude (decimal degrees)
    radius: Search radius in meters (defaults to 5000m/5km)
    connector_types: Optional list of specific connector types to filter by
                    (e.g., ["type2", "ccs", "tesla"])
    min_power: Minimum charging power in kW
    
Returns:
    List of charging stations with:
    - Location name and operator
    - Available connector types
    - Charging speeds
    - Number of charging points
    - Access restrictions
    - Other relevant metadata
. Input: {"latitude": {"title": "Latitude", "type": "number"}, "longitude": {"title": "Longitude", "type": "number"}, "radius": {"default": 5000, "title": "Radius", "type": "number"}, "connector_types": {"default": null, "items": {"type": "string"}, "title": "Connector Types", "type": "array"}, "min_power": {"default": null, "title": "Min Power", "type": "number"}}
- search_category (server: osm-mcp-server): 
Search for specific types of places within a defined geographic area.

This tool allows targeted searches for places matching specific categories within
a rectangular geographic region. It's particularly useful for filtering places by type
(restaurants, schools, parks, etc.) within a neighborhood or city district. Results include
complete location details and metadata about each matching place.

Args:
    category: Main OSM category to search for (e.g., "amenity", "shop", "tourism", "building")
    min_latitude: Southern boundary of search area (decimal degrees)
    min_longitude: Western boundary of search area (decimal degrees)
    max_latitude: Northern boundary of search area (decimal degrees)
    max_longitude: Eastern boundary of search area (decimal degrees)
    subcategories: Optional list of specific subcategories to filter by (e.g., ["restaurant", "cafe"])
    
Returns:
    Structured results including:
    - Query parameters
    - Count of matching places
    - List of matching places with coordinates, names, and metadata
. Input: {"category": {"title": "Category", "type": "string"}, "min_latitude": {"title": "Min Latitude", "type": "number"}, "min_longitude": {"title": "Min Longitude", "type": "number"}, "max_latitude": {"title": "Max Latitude", "type": "number"}, "max_longitude": {"title": "Max Longitude", "type": "number"}, "subcategories": {"default": null, "items": {"type": "string"}, "title": "Subcategories", "type": "array"}}
- find_schools_nearby (server: osm-mcp-server): 
Locate educational institutions near a specific location, filtered by education level.

This specialized search tool identifies schools, colleges, and other educational institutions
within a specified distance from a location. Results can be filtered by education level
(elementary, middle, high school, university, etc.). Essential for families evaluating
neighborhoods or real estate purchases with education considerations.

Args:
    latitude: Center point latitude (decimal degrees)
    longitude: Center point longitude (decimal degrees)
    radius: Search radius in meters (defaults to 2000m/2km)
    education_levels: Optional list of specific education levels to filter by
                    (e.g., ["elementary", "secondary", "university"])
    
Returns:
    List of educational institutions with:
    - Name and type
    - Distance from search point
    - Education levels offered
    - Contact information if available
    - Other relevant metadata
. Input: {"latitude": {"title": "Latitude", "type": "number"}, "longitude": {"title": "Longitude", "type": "number"}, "radius": {"default": 2000, "title": "Radius", "type": "number"}, "education_levels": {"default": null, "items": {"type": "string"}, "title": "Education Levels", "type": "array"}}
- find_parking_facilities (server: osm-mcp-server): 
Locate parking facilities near a specific location.

This tool finds parking options (lots, garages, street parking) near a specified location.
Results can be filtered by parking type and include capacity information where available.
Useful for trip planning, city navigation, and evaluating parking availability in urban areas.

Args:
    latitude: Center point latitude (decimal degrees)
    longitude: Center point longitude (decimal degrees)
    radius: Search radius in meters (defaults to 1000m/1km)
    parking_type: Optional filter for specific types of parking facilities
                ("surface", "underground", "multi-storey", etc.)
    
Returns:
    List of parking facilities with:
    - Name and type
    - Capacity information if available
    - Fee structure if available
    - Access restrictions
    - Distance from search point
. Input: {"latitude": {"title": "Latitude", "type": "number"}, "longitude": {"title": "Longitude", "type": "number"}, "radius": {"default": 1000, "title": "Radius", "type": "number"}, "parking_type": {"default": null, "title": "Parking Type", "type": "string"}}
- find_nearby_places (server: osm-mcp-server): 
Discover points of interest and amenities near a specific location.

This tool performs a comprehensive search around a geographic point to identify
nearby establishments, amenities, and points of interest. Results are organized by
category and subcategory, making it easy to find specific types of places. Essential
for location-based recommendations, neighborhood analysis, and proximity-based decision making.

Args:
    latitude: Center point latitude (decimal degrees)
    longitude: Center point longitude (decimal degrees)
    radius: Search radius in meters (defaults to 1000m/1km)
    categories: List of OSM categories to search for (e.g., ["amenity", "shop", "tourism"]).
            If omitted, searches common categories.
    limit: Maximum number of total results to return
    
Returns:
    Structured dictionary containing:
    - Original query parameters
    - Total count of places found
    - Results grouped by category and subcategory
    - Each place includes name, coordinates, and associated tags
. Input: {"latitude": {"title": "Latitude", "type": "number"}, "longitude": {"title": "Longitude", "type": "number"}, "radius": {"default": 1000, "title": "Radius", "type": "number"}, "categories": {"default": null, "items": {"type": "string"}, "title": "Categories", "type": "array"}, "limit": {"default": 20, "title": "Limit", "type": "integer"}}
- geocode_address (server: osm-mcp-server): 
Convert an address or place name to geographic coordinates with detailed location information.

This tool takes a text description of a location (such as an address, landmark name, or
place of interest) and returns its precise geographic coordinates along with rich metadata.
The results can be used for mapping, navigation, location-based analysis, and as input to
other geospatial tools.

Args:
    address: The address, place name, landmark, or description to geocode (e.g., "Empire State Building", 
            "123 Main St, Springfield", "Golden Gate Park, San Francisco")
    
Returns:
    List of matching locations with:
    - Geographic coordinates (latitude/longitude)
    - Formatted address
    - Administrative boundaries (city, state, country)
    - OSM type and ID
    - Bounding box (if applicable)
    - Importance ranking
. Input: {"address": {"title": "Address", "type": "string"}}
"""

    messages = [{"role": "user", "content": prompts}]
    prompts = [tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False) for m in messages]

    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attn = encoded["attention_mask"].to(device)

    x, think_span, tool_span, gen_end = generate_think_then_tool_but_decode_tool_first(
        model=model,
        prompt_ids=input_ids,
        attention_mask=attn,
        steps=128,
        think_budget=128,
        tool_budget=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        tokenizer=tokenizer,
        suppress_eos_eot=True,
        save_intermediate=True,
        output_file="denoise_log_128_128_128.txt",
    )

    # 解码最终结果（只取生成区）
    gen_ids = x[0, input_ids.shape[1]:gen_end].detach().tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    print(text)
    print("-" * 80)

    # 也可以单独切出 THINK / TOOL 段（更方便调试）
    think_ids = x[0, think_span[0]:think_span[1]].detach().tolist()
    tool_ids = x[0, tool_span[0]:tool_span[1]].detach().tolist()
    #print("[THINK SLOT]")
    print(tokenizer.decode([tid for tid in think_ids if tid != 126336], skip_special_tokens=False))
    #print("-" * 40)
    #print("[TOOL SLOT]")
    print(tokenizer.decode([tid for tid in tool_ids if tid != 126336], skip_special_tokens=False))


if __name__ == "__main__":
    main()