```
sudo pip3 install tensorflowjs ipython
```

```
tensorflowjs_converter --input_format=tf_hub 'https://tfhub.dev/google/compare_gan/model_9_celebahq128_resnet19/1' /tmp/compare_gan
```

(signatures)

```
tensorflowjs_converter --input_format=tf_hub 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2' tfjs_gestures/mobilenetv2_100_224
```

https://github.com/tensorflow/tfjs-converter/pull/242

```
import tensorflow as tf

with open("tensorflowjs_model.pb", "rb") as fin:
    data = fin.read()

gd = tf.GraphDef.FromString(data)
constants = [node for node in gd.node if node.op == 'Const']
for field in  ('half_val', 'float_val', 'double_val', 'int_val', 'string_val',
               'scomplex_val', 'int64_val', 'bool_val', 'resource_handle_val',
               'variant_val', 'uint32_val', 'uint64_val'):
    const.attr["value"].tensor.ClearField(field)
data = gd.SerializeToString()
print(len(data))
with open("tensorflowjs_model.pb", "wb") as fout:
    fout.write(data)
```
   
```
wget https://cdn.jsdelivr.net/npm/@tensorflow/tfjs
```

```
sudo npm install -g browser-sync
```

```
browser-sync start -s -f . --no-notify --host 0.0.0.0 --port 8080
```
