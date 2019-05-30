from pyecharts import Pie
'''
原训练集饼图
'''

attr = ['NEGATIVE','EFFECT','MECHANISM','ADVICE','INT']
v1 = [23772,1687,1319,826,188]
pie = Pie('训练集',title_pos = 'center')
pie.add(
        '',attr,v1,                 #''：图例名（不使用图例）
        radius = [40,75],           #环形内外圆的半径
        is_label_show = True,       #是否显示标签
        label_text_color = None,    #标签颜色
        legend_orient = 'vertical', #图例垂直
        legend_pos = 'left',
        )
pie.render()

# '''
# 原测试集饼图
# '''
#
# attr = ['NEGATIVE','EFFECT','MECHANISM','ADVICE','INT']
# v1 = [4782,360,302,221,96]
# pie = Pie('测试集',title_pos = 'center')
# pie.add(
#         '',attr,v1,                 #''：图例名（不使用图例）
#         radius = [40,75],           #环形内外圆的半径
#         is_label_show = True,       #是否显示标签
#         label_text_color = None,    #标签颜色
#         legend_orient = 'vertical', #图例垂直
#         legend_pos = 'left',
#         )
# pie.render()
