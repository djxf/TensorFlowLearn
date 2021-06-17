#
#
#
#
#
#


#经典的Min-Max搜索

#Nega-Max搜索
all_move = 999
def NegaMax(depth) {
    if (depth == 0) {
        return evalute(side_to_move)
    }
    max = -9999
    for i in range(all_move):
        score = -NegaMax(depth - 1)
        if (score > max) {
            max = score
        }
    return max
}


随机着法的自对弈
