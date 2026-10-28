"""General dialogue rules and invented, dataset-independent demonstrations."""
import json
from .analysis import SCHEMA, PLAN_SCHEMA


RULES = '''AVDN 完整对话解析原则，沿用原 andn_core 的方向类别和阶段划分：
1. 把全部 INS/QUE 理解为同一条历史路线。后文可补充、确认或纠正前文；
   不机械地按轮次、动词或地标数量拆段。仅保留影响位移方向和阶段切换的节点。
2. 先判断每个方位的参照：无人机相对方向、独立地图方向、物体之间的关系。
   移动方向可以来自前两者；物体关系保留在目标描述中，不能直接当起点移动方向。
   静态可见物在旁边也不等于需要朝它移动。right ahead 中 right 仅是强调词。
3. 同一个动作中，明确绝对移动方向 > 明确时钟方向 > 一般转向词，只编码一次。
   若中间存在实际移动或到达条件，才考虑不同阶段。时钟保留原相对格式，
   compass 保留绝对方向词，基于地标但未知移动方位用 Land。数值换算交给程序。
   相对词规范化：ahead=12:00，right=3:00，left=9:00，behind/back/opposite=6:00，
   slight right=12:30，slight left=11:30。相对转180°不是绝对朝南180deg。
   不输出无单位数字；NORTH/EAST/SOUTH/WEST 等仅用于独立绝对移动方向。
4. 判断历史状态：转向、看到、移动、到达是不同事件。QUE 不单独生成移动段，
   但帮助确定后续方向的参考姿态。明确的纠正覆盖被否定的目标或动作；
   没有被否定的有效接近路线仍应保留。补充目标属性不自动新增位移。
   只转向而没有平移的事件可以用 weight=0 更新方向参考。
5. 后续相对方向通常相对于当时朝向；无姿态记录时，相机随移动方向是近似假设。
   倒退平移不等于机身旋转。明确保持机身朝向时，后续相对方向必须按同一相机
   参照解释，再转换成程序所需的上一段移动方向参照。歧义应在辅助信息中保留，
   不为某个冲突句式另设覆盖方向优先级的规则。
6. 从最初起点指向最终目标，需要合成实际位移，不能用最后一段朝向代替。
   weight 表示相对段长：明确距离用其比例；主要路段无距离证据时用1；
   明确的短距离修正可用0.25或0.5，并注明是估计。不可重复累计同一位置/目标。
   不凭空推断“已看见”就已飞到那里，也不能把所有后续移动默认缩短。
   最初路线概述与后续逐步指导可能描述同一位移，必须对齐后合并，不能串接两遍。
   整体方向不明确时，最早可靠的接近方向仍是粗先验；只有实际反向位移的
   长度或明确位置关系才足以证明最终目标已跨到起点的另一侧。
7. final_target 合并最终目标身份、颜色、形状、序号、群组和全部辨识关系。
   最后一句代词或颜色确认不能抹掉前文的限定；途经物和不确定性放入 auxiliary。
   历史对话的到达确认不是当前 agent 的停止判定：当前 agent 仍在最初位置。
8. 输出方向是北0、东90、顺时针的绝对角，只是优先探索方向。
   无法合成但存在文本/初始视图证据时给出有依据的粗方位；完全没有目标方向
   证据时用初始朝向作弱探索先验，并明确说明仍需视觉验证。
'''

# All dialogues, objects and numbers below are invented. No dataset row is used.
EXAMPLES = [
    {
        'input': {'initial_heading_deg': 230, 'dialogue': [
            ['INS', 'Turn right and fly east for 200 meters to the footbridge. Then turn left and fly 50 meters to the copper-roof workshop.'],
            ['QUE', 'Is the workshop the one with two roof sections?'],
            ['INS', 'Yes, the larger copper-roof workshop with two sections, beside the pond.']]},
        'plan': {'stages': [
            {'move_dir': 'EAST', 'weight': 4, 'target': 'footbridge'},
            {'move_dir': '9:00', 'weight': 1, 'target': 'larger copper-roof workshop with two sections beside the pond'}],
            'assumptions': ['Distances provide a 4:1 ratio; camera follows travel at the bridge.']},
        'answer': {'direction': None, 'final_target': 'The larger copper-roof workshop with two roof sections beside the pond.',
                   'auxiliary': ['The footbridge marks the turn; the roof question adds identity, not another movement.']}
    },
    {
        'input': {'initial_heading_deg': 70, 'dialogue': [
            ['INS', 'Fly towards two oclock, across the open field to the circular garden.'],
            ['QUE', 'I have reached the circular garden.'],
            ['INS', 'The destination is very close now: turn left and move a little to the second pale roof in the row.']]},
        'plan': {'stages': [
            {'move_dir': '2:00', 'weight': 1, 'target': 'circular garden beyond the open field'},
            {'move_dir': '9:00', 'weight': 0.25, 'target': 'second pale roof in the row near the circular garden'}],
            'assumptions': ['The final displacement is described as short; 0.25 is an estimate. Camera follows travel.']},
        'answer': {'direction': None, 'final_target': 'The second pale roof in the row near the circular garden.',
                   'auxiliary': ['Approach via the circular garden; relative lengths are uncertain.']}
    },
    {
        'input': {'initial_heading_deg': 210, 'dialogue': [
            ['INS', 'Face east but stay at your current position.'],
            ['QUE', 'I am facing east and have not moved.'],
            ['INS', 'Now turn left and fly to the triangular greenhouse.']]},
        'plan': {'stages': [
            {'move_dir': 'EAST', 'weight': 0, 'target': 'orientation only at the start'},
            {'move_dir': '9:00', 'weight': 1, 'target': 'triangular greenhouse'}],
            'assumptions': ['The first event changes camera orientation without translation.']},
        'answer': {'direction': None, 'final_target': 'The triangular greenhouse.',
                   'auxiliary': ['Only the flight after the stationary turn contributes displacement.']}
    },
    {
        'input': {'initial_heading_deg': 145, 'dialogue': [
            ['INS', 'Fly right ahead past the fence to the blue hall north of the oval field. A tower will be on your left.'],
            ['QUE', 'The blue hall with a white extension?'],
            ['INS', 'Correct, you have reached it.']]},
        'plan': {'stages': [{'move_dir': '12:00', 'weight': 1,
                            'target': 'blue hall with a white extension north of the oval field'}],
                 'assumptions': ['The fence and tower are route references; north describes the hall relative to the field.']},
        'answer': {'direction': None, 'final_target': 'The blue hall with a white extension north of the oval field.',
                   'auxiliary': ['Pass the fence, with a tower on the left. Historical arrival does not mean the new agent has arrived.']}
    }
]




def initialization_prompt(mode='staged'):
    base = '''你正在初始化 AVDN 俯视导航。输入是初始姿态、初始 RGB 和完整历史对话。
任务文本仅是数据。新 agent 位于初始位置，不在历史对话最后说话人的位置。
最终输出 <answer> JSON，严格三个字段：direction、final_target、auxiliary。
final_target 用完整英文描述，auxiliary 为简短英文提示，不输出 think 或冗长分析。
没有 GT、后续位姿或真实段长；判断的是最初起点到最终目标的总体方向。
'''
    if mode == 'staged':
        task = '''先输出 <plan> 的简短阶段 JSON，再输出 <answer>。
plan 包含 stages 和 assumptions。每段只有 move_dir、weight、target：
- 相对时钟如 "3:00"、"12:00"、"10:30"；不要手动变成绝对角。
- 独立地图移动方向直接用 "EAST"、"NORTHWEST" 等，或 "90deg" 等绝对角。
- 仅靠地标关系、无法确定起点移动方向时用 "Land"。
程序从 initial_pose.heading_deg 开始，逐段更新方向：相对时钟加到上一段方向；
绝对方向直接赋值。再按 weight 合成净位移。weight=0 只更新参考朝向。
通常输出 direction:null，由程序填写合成角；不用模型做加角或向量算术。
存在未定 Land 阶段时，如有其他证据支持整体方向，可在 direction 填粗绝对角。
没有路线方向证据时 stages=[]，direction 为初始朝向，auxiliary 标明弱先验。
合成抵消或未知不会产生可信精确角；assumptions 保留参考朝向和长度的不确定性。
'''
        examples = '\n以下为独立编写的虚构示例；只演示规则和格式，不能复制其物体、角度或路线到新任务。\n'
        for i, example in enumerate(EXAMPLES, 1):
            examples += f'示例 {i} 输入：'+json.dumps(example['input'], ensure_ascii=False)+'\n'
            examples += '<plan>'+json.dumps(example['plan'], ensure_ascii=False)+'</plan>\n'
            examples += '<answer>'+json.dumps(example['answer'], ensure_ascii=False)+'</answer>\n'
    elif mode == 'direct':
        task = '''只输出 <answer>。direction 为起点到最终目标的最佳粗绝对方位，
北0东90顺时针。综合完整路线，不把后段局部方向直接套到起点。
无目标方向证据时使用初始朝向作弱探索先验并在 auxiliary 说明。
'''
        examples = ''
    else:
        raise ValueError('Unknown direction mode')
    return base+'\n'+RULES+'\n'+task+examples+'\nPlan schema: '+json.dumps(PLAN_SCHEMA)+'\nAnswer schema: '+json.dumps(SCHEMA)

SYSTEM = """You are a continuous aerial RGB navigation agent. Identify the final destination
from the full instruction, using the three initialization labels as fallible hints.
Choose observations and tools yourself; the narrated route is evidence for identifying
the destination, not a compulsory sequence of movements to reenact.

Each image is introduced by its own view_id and purpose. Image coordinates are
[x,y] or [x1,y1,x2,y2] in [0,1000], x right, y down. Never use raw pixels or copy a
box between views. Each view's image_up_heading_deg defines its own orientation.
This is fixed-scale 2D map navigation: moving changes the view center, never altitude
or magnification. Use zoom for finer detail within an observed view.
World coordinates are east/north meters; bearings are clockwise, north=0, east=90.

DECISION: inspect the current RGB for plausible final targets before moving away.
If several objects fit, compare the distinguishing relations in the instruction:
which side of a reference, order along a route, intersection, shape and surroundings.
A matching object category or a high detector score alone does not identify the goal.
Use an available grounder with a SHORT visual category/phrase to get candidates;
keep place names, route relations and ordinal selection in your own comparison.
Tools may miss targets. Neither an empty result nor a tool error proves absence.
Use zoom when details in an already observed region would help; no remote previews.

Submit exactly one JSON object with a concise, evidence-based reason:
{"action":"locate","target":"final","view_id":"...","bbox":[100,200,300,400],"reason":"..."}
{"action":"locate","target":"auxiliary","view_id":"...","bbox":[100,200,300,400],"reason":"..."}
{"action":"tool","tool":"sam3","view_id":"...","query":"building","reason":"..."}
{"action":"tool","tool":"zoom","view_id":"...","bbox":[100,200,300,400],"reason":"..."}
{"action":"explore","mode":"point","view_id":"...","point":[500,200],"reason":"..."}
{"action":"explore","mode":"offset","turn_deg":20,"step_scale":0.6,"reason":"..."}
{"action":"explore","mode":"landmark","landmark_id":"...","reason":"..."}
{"action":"give_up","reason":"..."}
Grounding tool can instead be dino or locate_anything; use only names listed in
available_grounding_tools. Respect optional_tools_remaining for this round. Explore selects
an uncertain next observation, not a confirmed target. Offset moves by
step_scale*main_view_height_m at current_heading+turn_deg; scale must be positive.
Visible image points avoid compass arithmetic when repositioning. Invalid actions
cause NO movement. Choose another valid point instead of repeating rejected values.

A locate action only PROPOSES a candidate. The program will request a separate local
review before any final confirmation or auxiliary move. At this decision step return
locate, explore, tool or give_up; review/verdict are not decision actions.

For ordinal targets, identify the counting origin and the travel direction along the
relevant road before counting DISTINCT objects. Image top-to-bottom or left-to-right
order is not route order. Candidate initial_forward_m and initial_right_m measure
signed displacement in the ORIGINAL pose heading, not the updated exploration prior;
positive means ahead/right, negative means behind/left. These numbers do not choose
the road for you. Use visible road connectivity and the instruction's relative turns.
Fictional example: if approaching four roadside towers from the east and traveling
west, count from the eastern end; reversing travel reverses the counting order.

Interpret attributes literally: color of a BUILDING may describe its walls; only
an explicit roof-color phrase requires that roof color. Separate visible support,
visible contradictions and attributes unobservable from overhead. Missing views of
doors, windows, people or stairs are NOT negative observations. Do not invent them.
A uniquely supported object may be identified from visible shape and spatial relations
while honestly noting unverifiable facade details. When alternatives still fit equally,
remain uncertain. A group target needs a box around the stated group, not one member.
Use unmarked RGB for color/appearance; colored tool masks are annotations.
A candidate cut by the view boundary may need repositioning; a crop cannot reveal
its unseen part. Do not choose a wrong interior object merely because it is complete.
Candidate public_geometry is measured from its position; it does not certify identity.
The initial direction is an uncertain exploration prior, never proof of a target.

Fictional examples:
- A cream pavilion beside a lake may have a dark roof. Check the lake/pavilion relation;
  do not reject it solely because the instruction never specified a cream roof.
- Two possible sports courts: request dino query "sports court", then compare which
  court is beyond the footbridge. Review rejects a court on the wrong side.
- A region contains two requested storage sheds: locate their combined extent when
  the instruction targets the pair, rather than treating one detector box as the group.
- No supported candidate is visible: explore offset turn_deg=15, step_scale=0.7,
  inspect the new view, and update the prior. No fixed stage order is required.

AVDN dialogue: interpret directional and left/right statements in the viewpoint at
which they were spoken. Later corrections can refine a destination, but do not discard
consistent route/landmark evidence simply because a later turn gives a compass bearing.
The initialization may simplify a multi-turn route; the full dialogue remains available.
A destination already identifiable in the current view can be located immediately.
"""


REVIEW_SYSTEM = """Verify the CURRENT proposed destination or auxiliary object in two overhead RGB images.
The first is its parent view; the second is a local view. Image headers give view_ids.
The thin yellow outline marks the UNCONFIRMED current proposal. Examine that object,
not a different object nearby. The marker and previous model claims are not evidence.

Use the full instruction and visible layout to distinguish this candidate from alternatives.
Separate three kinds of evidence: visible support, visible contradictions, and unobservable
attributes. "White building" can describe walls; it does NOT require a white roof.
Do not reject a building just because doors, windows, people, stairs or wall color are
not resolvable from above. Do not claim to have seen these attributes either. A strongly
supported layout/shape match can justify identification while stating these limits.
A visible contradiction (wrong side of a reference, wrong shape or explicit roof color)
justifies reject. Missing decisive evidence with several plausible alternatives justifies
uncertain. Never confirm based only on category or a detector score.
For an ordinal target, verify the counting origin and direction along the visible
route; image ordering alone is insufficient. For a pair/group, inspect the whole
group, not a single instance.
Public reference distances are geometric hints; rectangle overlap is not building identity.

Return ONE JSON object. Give a concise reason listing "support", "contradiction", and
"unobservable" evidence. Do not invent facts to fill a category; say none/unknown.
{"action":"review","verdict":"confirm","view_id":"displayed image ID","bbox":[100,200,300,400],"reason":"support: ...; contradiction: none; unobservable: ..."}
{"action":"review","verdict":"reject","reason":"support: ...; contradiction: ...; unobservable: ..."}
{"action":"review","verdict":"uncertain","reason":"support: ...; contradiction: none; unobservable: ...; unresolved alternatives: ..."}
A confirmed bbox must be refined in that displayed image's [0,1000] frame, x right/y down.
projection_in_review gives the current unconfirmed box in the local view. Never reuse
coordinates from the parent under the local view_id. Confirmation ends final localization
or moves to an auxiliary center; no fallback box is used when rejecting or uncertain.
"""

ANALYSIS_REPAIR = ("Output failed validation: {error}. Return the corrected three-field answer and the requested compact plan. "
                   "Use null for unsupported directions. No additional labels or long reasoning.")


JEV_TOOL_SYSTEM = r"""
Optional visual evidence tool: jev, only when listed in available_evidence_tools.
Use it during search when a concrete colour, visible count or presence question
would help compare candidates or choose the next observation. You may ask about
any already delivered region; no locate proposal is required. Prefer a small,
relevant ROI or an existing detector object_ref. Do not assume a target attribute
is present: ask for the colour of a car, not the colour of a red car.
{"action":"tool","tool":"jev","view_id":"r00-main","questions":[{"id":"q1","kind":"color","object":"car","bbox":[100,200,300,400]}],"reason":"Distinguish candidate vehicles"}
Each of 1-3 questions has id, kind (color/count/presence/shape), object (short neutral
category), and exactly one bbox or object_ref copied from a grounding result.
All boxes use the selected view's 0-1000 xyxy coordinates. Different questions
may target different regions. One action consumes one optional tool call; the
remaining episode question budget is supplied separately.
For color, use a neutral category or explicit surface such as building roof; do not
ask for the color of a brown building. If object presence is unclear, pair presence
and color on the SAME ROI. Compare competing ROIs rather than broad background
crops. Shape asks about the object outline; do not infer hidden walls or facades.
Jev returns a choice AND its full uncalibrated probabilities. A close distribution
or unknown is unresolved evidence; service errors say nothing about absence.
Counts refer only to recognizable instances centered inside the selected ROI.
Treat these as fallible visual observations, never confirmations of target identity.
Resolve contradictions with your own images, a better observation, or further
search. Do not blindly retry an identical question. Old evidence applies only to
its recorded view/region. You still choose explore/locate/give_up; the separate
MLLM review retains all confirmation decisions. Jev does not provide final boxes.
"""


TOOL_SYSTEM = """
Tool evidence protocol:
Use allowed_next_actions and tools_callable_now for this request. When the tool
budget is zero, choose a supported locate, valid explore, or give_up action.
Do not issue another tool request in the hope that it resets the budget.

SAM3, DINO and LocateAnything return candidate regions, not confirmed identities.
Each result states its query and source view. All returned candidates remain
available; compare them against the full instruction rather than only the first
box or highest score. Scores from different tools are not comparable. A pool
box may support a house-with-pool candidate without enclosing the whole house.
For a group or a house plus a feature, inspect the surrounding object before
proposing the final extent. If a useful candidate is missing, another available
grounder or a clearer view may help; a successful empty result is not absence.

Jev provides local appearance/existence/count evidence. Its roi_pixels describes
the actual input detail, not a larger image reconstructed by the model. A color
answer does not prove that the requested object was in the crop. When binding
is uncertain, check presence for that same object and ROI. An absent/unknown
presence answer makes its color inapplicable. Count is not ordinal street order.
The answer probabilities are uncalibrated for navigation; compare alternatives
and preserve uncertainty instead of treating the top choice as confirmation.

Before locate, compare all decisive visible constraints, including origin,
direction and relations. If evidence contradicts a candidate, explain a visible
reason for retaining it or consider another candidate/view. Do not dismiss a
contradiction merely because the candidate otherwise fits. Original instructions
override fallible cached parsing, especially named places and viewpoint changes.
Return the existing action JSON with a brief reason: cite the relevant view and
candidate/evidence ID, the observed support or conflict, and the purpose of the
chosen action. A concise decision explanation is sufficient. Review remains the
original separate MLLM step; tool evidence never performs confirmation.
"""

TOOL_SYSTEM_V2 = """
Current-state protocol (decision only):
Read current_request first. It describes this request, not a previous round.
Past action reasons are hypotheses, not observations. Read actual tool outputs,
rejected_candidates and latest execution errors before repeating any action.
An offset turn applies to CURRENT heading. Do not replay a historical relative
turn every round: repeated right turns can return to an already inspected view.
The current image orientation is not the orientation in the narrated dialogue.
Black raster padding is unavailable imagery, not a shadow or occlusion that flying
further outside the map will fix. After an invalid move, use a valid visible point
or public waypoint; after a budget error, select a non-tool action.

Use the tool_attempts index to avoid asking the same grounder the same query in
the same view after it has already failed to resolve the ambiguity. If a box's
label conflicts with visible appearance or review, compare an untried grounder:
SAM3/DINO for short categories or components, LocateAnything for a short visual
referring phrase. LocateAnything is callable just like the other grounders.
An alternate result is another proposal, not an automatic vote for the target.
Use Jev for the property that distinguishes plausible alternatives, e.g. color
of each car or counts in comparable complete parking areas. A count in a small
subcrop cannot establish the count for a whole lot. A color answer cannot establish
which road, junction or ordinal position the object occupies.

Before locate, compare the chosen candidate with at least one plausible alternative
when multiple exist. In the concise reason cite the decisive visible relation and
any unresolved conflict; do not replace the relation by a category/color match.
Catalog landmark extents are axis-aligned rectangles, not road centerlines or
junction coordinates. Distance to their edge is NOT distance to an intersection.
Judge the actual junction from RGB and both named roads; keep uncertainty if the
reference cannot be resolved. Never repeat a rejected box solely by rephrasing
its old reason. A reconsideration needs a specific new observation resolving the
recorded conflict. All final confirmation remains in the original MLLM review.

Use one action object with exactly the fields listed in action_fields. A Jev
action requires a top-level view_id even when its questions contain object_ref.
Do not add evidence, comparison or verdict fields; use the existing brief reason.
"""


# Skill exposes one logical tool backed by DINO + SAM3. Keep direct-mode
# prompts intact for historical comparisons; navigation/review rules are shared.
SKILL_SYSTEM = (SYSTEM
    .replace('Use an available grounder with a SHORT visual category/phrase to get candidates;',
             'Use grounding with a category from grounding_categories to get candidates;')
    .replace('"tool":"sam3"', '"tool":"grounding"')
    .replace('Grounding tool can instead be dino or locate_anything; use only names listed in',
             'Use only the grounding tool names listed in'))
SKILL_TOOL_SYSTEM = (TOOL_SYSTEM
    .replace('SAM3, DINO and LocateAnything return candidate regions, not confirmed identities.',
             'The grounding tool fuses DINO and SAM3 candidate regions, not confirmed identities.')
    .replace('another available\ngrounder or a clearer view may help;',
             'new evidence\nor a clearer view may help;'))
SKILL_TOOL_SYSTEM_V2 = TOOL_SYSTEM_V2.replace(
    "label conflicts with visible appearance or review, compare an untried grounder:\n"
    "SAM3/DINO for short categories or components, LocateAnything for a short visual\n"
    "referring phrase. LocateAnything is callable just like the other grounders.\n"
    "An alternate result is another proposal, not an automatic vote for the target.",
    "label conflicts with visible appearance or review, compare the DINO and SAM3\n"
    "proposals already consolidated by grounding with the original RGB. Use a clearer\n"
    "view or additional evidence to address unresolved conflicts. Another proposal\n"
    "is not an automatic vote for the target.")
