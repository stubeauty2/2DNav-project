"""General dialogue rules and invented, dataset-independent demonstrations."""

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
