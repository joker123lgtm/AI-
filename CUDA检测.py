import tensorflow as tf

print("TensorFlow版本:", tf.__version__)
print("GPU设备列表:", tf.config.list_physical_devices('GPU'))
print("GPU是否可用:", tf.test.is_gpu_available())
print("GPU设备名称:", tf.test.gpu_device_name())

# 检查CUDA和cuDNN版本
build_info = tf.sysconfig.get_build_info()
print("CUDA版本:", build_info.get("cuda_version", "未找到"))
print("cuDNN版本:", build_info.get("cudnn_version", "未找到"))

# 测试GPU计算
if tf.config.list_physical_devices('GPU'):
    print("\n=== GPU测试 ===")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("GPU计算成功:")
        print(c.numpy())
else:
    print("未检测到GPU设备")