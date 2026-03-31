import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


MASK_ID = 126336
EOS_ID = 126081
EOT_ID = 126348


@dataclass
class PlanParseResult:
    text: str
    plan_json: Optional[dict]
    plan_end_pos: Optional[int]
    call_ids: List[str]
    valid: bool


# =============================
# 基础工具函数
# =============================
def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score
    but reduces generation quality. Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    预计算每一步需要转移多少 token。
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def decode_visible_with_masks(tokenizer, ids, mask_id):
    """
    将当前中间状态解码出来，mask 显示成 <MASK>，方便看日志。
    """
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

    return "".join(parts)


# =============================
# plan 解析逻辑
# =============================
def extract_json_block(text: str) -> Tuple[Optional[dict], Optional[int]]:
    """
    从文本中提取第一个顶层 JSON 对象。
    返回: (json_obj, end_char_pos_exclusive)
    """
    start = text.find("{")
    if start == -1:
        return None, None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate), i + 1
                except Exception:
                    return None, None

    return None, None


def parse_plan_from_text(text: str) -> PlanParseResult:
    """
    解析第一阶段输出的 plan，要求格式至少是：
    {
      "calls": [
        {"id": "c1", "tool": "...", "depends_on": [...], "goal": "..."}
      ]
    }
    """
    plan_json, plan_end = extract_json_block(text)

    if not isinstance(plan_json, dict):
        return PlanParseResult(text, None, None, [], False)

    calls = plan_json.get("calls")
    if not isinstance(calls, list):
        return PlanParseResult(text, None, None, [], False)

    call_ids = []
    for item in calls:
        if not isinstance(item, dict):
            return PlanParseResult(text, None, None, [], False)

        cid = item.get("id")
        tool = item.get("tool")

        if not isinstance(cid, str) or not isinstance(tool, str):
            return PlanParseResult(text, None, None, [], False)

        call_ids.append(cid)

    return PlanParseResult(text, plan_json, plan_end, call_ids, True)


def build_args_template(call_ids: List[str]) -> str:
    """
    根据 plan 动态生成第二阶段的 ARGS block 模板。
    """
    blocks = []
    for cid in call_ids:
        blocks.append(
            f'\n<ARGS call_id="{cid}">\n'
            + '{"arguments": {}}\n'
            + '</ARGS>'
        )
    return "".join(blocks)


# =============================
# prompt 构造
# =============================
def make_plan_prompt(query: str, tools_text: str) -> str:
    return f"""You are an intelligent assistant designed to help users accomplish tasks using tools.

You must solve the task in TWO STAGES.

Stage 1: output ONLY a JSON object with this schema:
{{
  "calls": [
    {{"id": "c1", "tool": "tool_name", "depends_on": [], "goal": "short goal"}}
  ]
}}

Rules for Stage 1:
1. Output valid JSON only.
2. Do not output any <ARGS> block.
3. Each call must contain id, tool, depends_on, goal.
4. Use only tools from the available list.
5. If multiple calls are needed, put them in the calls array.

User query:
{query}

Available tools:
{tools_text}
"""


def make_args_prompt(query: str, tools_text: str, plan_json: dict) -> str:
    plan_str = json.dumps(plan_json, ensure_ascii=False, indent=2)
    return f"""You are an intelligent assistant designed to help users accomplish tasks using tools.

The tool plan is already fixed. You MUST NOT modify tool names, ids, or dependencies.
Now perform Stage 2: generate one <ARGS> block for each call in the plan.

Output format:
<ARGS call_id="c1">
{{"arguments": {{...}}}}
</ARGS>

Requirements:
1. Output only <ARGS> blocks.
2. One block per call in the plan.
3. Respect dependencies in the plan when deciding arguments.
4. Do not repeat the plan JSON.
5. Do not add natural language explanation.

User query:
{query}

Available tools:
{tools_text}

Fixed plan:
{plan_str}
"""


# =============================
# 核心扩散解码：支持 active mask
# =============================
@torch.no_grad()
def generate_with_active_mask(
    model,
    prompt,
    active_gen_mask,
    attention_mask=None,
    steps=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=MASK_ID,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    save_intermediate=False,
    tokenizer=None,
    output_file="generation_process.txt",
    stage_name="stage",
):
    """
    active_gen_mask: Bool tensor, shape (B, gen_len)
      True  -> 当前阶段允许解码
      False -> 当前阶段冻结，不参与提交
    """
    bsz = prompt.shape[0]
    gen_len = active_gen_mask.shape[1]

    x = torch.full(
        (bsz, prompt.shape[1] + gen_len),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        full_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (bsz, gen_len),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )
    else:
        full_attention_mask = None

    prompt_index = (x != mask_id)
    gen_start = prompt.shape[1]

    # 哪些位置是当前阶段可以提交的
    active_full = torch.zeros_like(x, dtype=torch.bool)
    active_full[:, gen_start:] = active_gen_mask

    stage_mask_index = active_full & (x == mask_id)
    num_transfer_tokens = get_num_transfer_tokens(stage_mask_index[:, gen_start:], steps)

    for i in range(steps):
        mask_index = (x == mask_id)
        candidate_mask = mask_index & active_full

        if candidate_mask.sum().item() == 0:
            break

        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            if full_attention_mask is not None:
                attention_mask_ = torch.cat([full_attention_mask, full_attention_mask], dim=0)
            else:
                attention_mask_ = None
            logits = model(x_, attention_mask=attention_mask_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x, attention_mask=full_attention_mask).logits

        if logits_eos_inf:
            logits[:, :, EOS_ID] = -torch.inf

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if confidence_eos_eot_inf:
            logits_with_noise[:, :, EOS_ID] = -torch.inf
            logits_with_noise[:, :, EOT_ID] = -torch.inf

        if remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        elif remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        # 只有当前 active 区域参与竞争
        x0_p = torch.where(candidate_mask, x0_p, torch.full_like(x0_p, -np.inf))
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(candidate_mask, x0_p, torch.full_like(x0_p, -np.inf))

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

        for j in range(confidence.shape[0]):
            k = int(num_transfer_tokens[j, i].item())
            if k <= 0:
                continue

            valid_count = int(torch.isfinite(confidence[j]).sum().item())
            k = min(k, valid_count)
            if k <= 0:
                continue

            _, select_index = torch.topk(confidence[j], k=k)
            transfer_index[j, select_index] = True

        x[transfer_index] = x0[transfer_index]

        if save_intermediate and tokenizer is not None:
            ids = x[0, gen_start:].detach().tolist()
            intermediate_text = decode_visible_with_masks(tokenizer, ids, mask_id)

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"[{stage_name}] Step {i+1}/{steps} transferred={int(transfer_index[0].sum().item())}\n")
                f.write(intermediate_text + "\n")
                f.write("-" * 60 + "\n")

    return x[:, gen_start:]


# =============================
# 两阶段逻辑：先 plan，再动态 args
# =============================
@torch.no_grad()
def generate_plan_then_args(
    model,
    tokenizer,
    query,
    tools_text,
    device="cuda",
    plan_steps=96,
    args_steps=96,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    save_intermediate=True,
    output_file="dynamic_decode_log.txt",
):
    """
    两阶段解码：
    1) 先生成 plan
    2) 从 plan 中解析 calls 数量
    3) 动态生成对应数量的 ARGS block
    4) 再生成 args
    """

    # -------------------------
    # Stage 1: 只解 plan
    # -------------------------
    plan_prompt = make_plan_prompt(query, tools_text)
    messages = [{"role": "user", "content": plan_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # 给 plan 一个生成画布
    plan_canvas_text = '{\n  "calls": []\n}' + "\n" * 8
    plan_canvas_ids = tokenizer(
        plan_canvas_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    plan_gen_len = plan_canvas_ids.shape[1]
    active_plan_mask = torch.ones((1, plan_gen_len), dtype=torch.bool, device=device)

    plan_tokens = generate_with_active_mask(
        model=model,
        prompt=input_ids,
        active_gen_mask=active_plan_mask,
        attention_mask=attention_mask,
        steps=plan_steps,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=MASK_ID,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        save_intermediate=save_intermediate,
        tokenizer=tokenizer,
        output_file=output_file,
        stage_name="plan",
    )

    plan_text = tokenizer.decode(plan_tokens[0], skip_special_tokens=True)
    parsed = parse_plan_from_text(plan_text)

    if not parsed.valid:
        raise ValueError(
            "Stage-1 plan parsing failed. The model did not produce a valid JSON plan.\n\n"
            f"Raw plan text:\n{plan_text}"
        )

    # -------------------------
    # Stage 2: 根据 calls 动态生成 ARGS block
    # -------------------------
    args_template = build_args_template(parsed.call_ids)

    args_prompt = make_args_prompt(query, tools_text, parsed.plan_json)
    messages = [{"role": "user", "content": args_prompt}]
    args_prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    encoded = tokenizer(
        args_prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    args_input_ids = encoded["input_ids"].to(device)
    args_attention_mask = encoded["attention_mask"].to(device)

    args_canvas_ids = tokenizer(
        args_template,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    args_gen_len = args_canvas_ids.shape[1]
    active_args_mask = torch.ones((1, args_gen_len), dtype=torch.bool, device=device)

    args_tokens = generate_with_active_mask(
        model=model,
        prompt=args_input_ids,
        active_gen_mask=active_args_mask,
        attention_mask=args_attention_mask,
        steps=args_steps,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=MASK_ID,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        save_intermediate=save_intermediate,
        tokenizer=tokenizer,
        output_file=output_file,
        stage_name="args",
    )

    args_text = tokenizer.decode(args_tokens[0], skip_special_tokens=True)

    return {
        "plan_text": plan_text,
        "plan_json": parsed.plan_json,
        "args_text": args_text,
        "call_ids": parsed.call_ids,
    }


# =============================
# main
# =============================
def main():
    device = "cuda"

    model_path = "/data/labshare/Param/llada/"
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    assert tokenizer.pad_token_id != MASK_ID

    query = "I'm planning a trip to the Forbidden City in Beijing, show me where to park and what's interesting to do within 1km distance?"

    tools_text = """
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

    result = generate_plan_then_args(
        model=model,
        tokenizer=tokenizer,
        query=query,
        tools_text=tools_text,
        device=device,
        plan_steps=96,
        args_steps=96,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        save_intermediate=True,
        output_file="dynamic_decode_log.txt",
    )

    print("===== RESULT =====")
    print(result)
 


if __name__ == "__main__":
    main()