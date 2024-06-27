## NBA球星数据集

1. **Year**: 年份，指的是数据所属的赛季年份，例如2019-2020赛季。
2. **Player**: 球员，指的是球员的姓名或唯一标识符。这通常是一个字符串，表示球员的名字，有时可能包括名和姓。
3. **Pos**: 位置，指的是球员在球场上的主要位置。NBA篮球运动员的位置通常包括以下几种：
   - C: 中锋（Center）
   - PF: 大前锋（Power Forward）
   - SF: 小前锋（Small Forward）
   - SG: 得分后卫（Shooting Guard）
   - PG: 控球后卫（Point Guard）
4. **Age**: 球员的年龄。
5. **Tm**: 球员所在的球队（可能需要查找对应的球队缩写对照表）。
6. **G**: 球员参加的比赛场次（Games Played）。
7. **MP**: 球员平均每场比赛的出场时间（Minutes Played）。
8. **FG**: 投篮命中数（Field Goals Made）。
9. **FGA**: 投篮出手数（Field Goals Attempted）。
10. **FG%**: 投篮命中率（Field Goal Percentage），计算方式为 FG / FGA。
11. **3P**: 三分球命中数（Three-Point Field Goals Made）。
12. **3PA**: 三分球出手数（Three-Point Field Goals Attempted）。
13. **3P%**: 三分球命中率（Three-Point Field Goal Percentage），计算方式为 3P / 3PA。
14. **2P**: 两分球命中数（Two-Point Field Goals Made），计算方式为 FG - 3P。
15. **2PA**: 两分球出手数（Two-Point Field Goals Attempted），计算方式为 FGA - 3PA。
16. **2P%**: 两分球命中率（Two-Point Field Goal Percentage），计算方式为 2P / 2PA。
17. **eFG%**: 有效投篮命中率（Effective Field Goal Percentage），考虑了三分球的额外价值，计算方式为 (FG + 0.5 * 3P) / FGA。
18. **FT**: 罚球命中数（Free Throws Made）。
19. **FTA**: 罚球出手数（Free Throws Attempted）。
20. **FT%**: 罚球命中率（Free Throw Percentage），计算方式为 FT / FTA。
21. **ORB**: 进攻篮板数（Offensive Rebounds）。
22. **DRB**: 防守篮板数（Defensive Rebounds）。
23. **TRB**: 总篮板数（Total Rebounds），计算方式为 ORB + DRB。
24. **AST**: 助攻数（Assists）。
25. **STL**: 抢断数（Steals）。
26. **BLK**: 盖帽数（Blocks）。
27. **TOV**: 失误数（Turnovers）。
28. **PF**: 犯规数（Personal Fouls）。
29. **PTS**: 总得分（Points Scored），计算方式为 FG * 2 + 3P * 3 + FT。