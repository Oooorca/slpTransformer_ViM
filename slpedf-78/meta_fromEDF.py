import pyedflib
from datetime import datetime, timedelta

# 输入EDF文件路径
edf_path = './ST_meta/telemetry/ST7242J0/ST7242JO-Hypnogram.edf'

# 输出txt文件路径
output_path = './ST_meta/telemetry/ST7242J0/info/ST7242J0.txt'

# 打开EDF文件
edf_file = pyedflib.EdfReader(edf_path)

# 打开输出文件
with open(output_path, 'w') as output_file:
    output_file.write("Onset\tDuration\tAnnotation\n")

    # 获取 recording_start_time
    recording_start_time = edf_file.getStartdatetime()

    # 获取注释通道的数据
    annotation_channel = edf_file.readAnnotations()

    # 循环解析annotation并生成txt内容
    for onset, duration, description in zip(annotation_channel[0], annotation_channel[1], annotation_channel[2]):
        onset_time = recording_start_time + timedelta(seconds=int(onset))

        # 输出到txt文件
        output_file.write(f"{onset_time.strftime('%Y-%m-%dT%H:%M:%S.%f')}	{duration}	{description}\n")

# 关闭EDF文件
edf_file.close()
