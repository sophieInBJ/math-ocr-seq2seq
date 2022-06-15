# 每次训练要改下NAME，与生成文件名和模型名称有关，习惯按日期
NAME = '0614'

# 字符集
STR_FEN = "0123456789+-×÷=().~[]/<>…涂又'%"
STR_DANWEI = "圈小时分钟秒千克亿公里kg厘米cm平方元角吨万毫升日月年立百半刻d顷斤t框"
STR_UNKNOW = '&' # 未知字符
CHAR_LIST = STR_FEN + STR_DANWEI + STR_UNKNOW

# 是否加载模型
RESUME = False 
ENCODER_RESUME = './saves/0508/Encoder-0508-epoch16.pth'
DECODER_RESUME = './saves/0508/Decoder-0508-epoch16.pth'

NUM_EPOCHS = 30

# 加载的数据
TRAIN_TXT = './train_data/train.txt'
TEST_TXT = './train_data/test.txt'

# 优化器超参
ENCODE_LEARNING_RATE = 1e-4
DECODE_LEARNING_RATE = 1e-4
BETA1 = 0.9

# 输入size
IMAGE_H = 64
IMAGE_W = 240
