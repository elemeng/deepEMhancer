import os
import tensorflow as tf
import h5py
from .ioUtils import loadVolIfFnameOrIgnoreIfMatrix


def load_model(checkpoint_fname, custom_objects=None, lastLayerToFreeze=None, resetWeights=False, nGpus=1):
  if custom_objects is None:
    __, codes = retrieveParamsFromHd5(checkpoint_fname,  [], ['code/custom_objects'])
    if codes is None:
      from devel_code.trainNet.defaultNet import getCustomObjects
      print("WARNING, no custom obtects in model, using default")
      custom_objects = getCustomObjects()
    else:
      custom_objects= codes["custom_objects"]

  import shutil

  # Handle Keras 3 requirement for .h5 extension
  checkpoint_path = checkpoint_fname
  temp_file = None
  if checkpoint_fname.endswith('.hd5'):
    # Check if we're using Keras 3
    keras_version = None
    try:
      import keras
      keras_version = keras.__version__
    except ImportError:
      pass

    if keras_version and keras_version.startswith('3.'):
      # Create a temporary symlink or copy with .h5 extension for Keras 3
      import tempfile
      temp_dir = tempfile.gettempdir()
      temp_file = os.path.join(temp_dir, os.path.basename(checkpoint_fname).replace('.hd5', '.h5'))
      if not os.path.exists(temp_file):
        shutil.copy2(checkpoint_fname, temp_file)
      checkpoint_path = temp_file

  if int(tf.__version__.split(".")[0]) > 1:
    from tensorflow.keras.models import load_model
  else:
      from keras.models import load_model

  if nGpus>1:
    devices_names = list(map(lambda x:":".join( x.name.split(":")[-2:]), tf.config.list_physical_devices('GPU')))
    mirrored_strategy = tf.distribute.MirroredStrategy(devices= devices_names )
    with mirrored_strategy.scope():
      model = load_model(checkpoint_path, custom_objects=custom_objects )
  else:
      model = load_model(checkpoint_path, custom_objects=custom_objects )

  if lastLayerToFreeze is not None:
    layerFound= False
    for layer in model.layers:
      layer.trainable=False
      if layer.name.startswith(lastLayerToFreeze):
        layerFound=True
        break
    assert layerFound is True, "Error, %s not found in the model"%lastLayerToFreeze

  if resetWeights:
    print("Model reset")
    for i, layer in enumerate(model.layers):
      initializers = []
      if hasattr(layer, 'kernel_initializer'):
        initializers += [ lambda : layer.kernel_initializer(shape=model.layers[i].get_weights()[0].shape) ]
      if hasattr(layer, 'bias_initializer') and layer.bias is not None:
        initializers += [ lambda : layer.bias_initializer(shape=model.layers[i].get_weights()[1].shape) ]
      if len(initializers)>0:
        model.layers[i].set_weights( [f() for f in initializers ]  )
  return model


def getInputCubeSize(model):
  # Try to get input shape from the first layer
  input_layer = model.layers[0]

  # Keras 3: Try to get from input_shape property
  try:
    if hasattr(input_layer, 'input_shape') and input_layer.input_shape is not None:
      shape = input_layer.input_shape
      if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return shape[1]
  except (AttributeError, TypeError):
    pass

  # Keras 3: Try to get from output_shape property
  try:
    if hasattr(input_layer, 'output_shape') and input_layer.output_shape is not None:
      shape = input_layer.output_shape
      if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return shape[1]
  except (AttributeError, TypeError):
    pass

  # Keras 3: Try to get from batch_input_shape property
  try:
    if hasattr(input_layer, 'batch_input_shape') and input_layer.batch_input_shape is not None:
      shape = input_layer.batch_input_shape
      if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return shape[1]
  except (AttributeError, TypeError):
    pass

  # Keras 3: Try to get from config
  try:
    config = input_layer.get_config()
    if 'batch_shape' in config and config['batch_shape'] is not None:
      shape = config['batch_shape']
      if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return shape[1]
    elif 'shape' in config and config['shape'] is not None:
      shape = config['shape']
      if isinstance(shape, (list, tuple)) and len(shape) >= 1:
        return shape[0]
  except (AttributeError, TypeError, KeyError):
    pass

  # Try from model input shape
  try:
    if hasattr(model, 'input_shape') and model.input_shape is not None:
      shape = model.input_shape
      if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return shape[1]
  except (AttributeError, TypeError):
    pass

  raise ValueError("Unable to determine input cube size from model")


def retrieveParamsFromHd5(fname, paramsList, codeList):
  '''
  Example:

   retrieveParamsFromHd5(kerasCheckpointFname, ['configParams/*', 'configParams/NNET_INPUT_SIZE'], ['code/normFun', 'code/*'])

  :param fname:
  :param paramsList:
  :param codeList:
  :return:

  '''
  args= {}
  codes={}
  try:
    with h5py.File(fname,'r') as h5File:
#      print(h5File.keys())
      for paramName in paramsList:
        if paramName.endswith("*"):
          paramName=paramName.replace("/*", "")
          for putativeKey in h5File[paramName]:
            param=h5File[paramName+'/'+putativeKey][0]
            args[putativeKey]= param
        else:
          args[os.path.basename(paramName)] = h5File[paramName][0]

      env= globals()
      for codeName in codeList:
        if codeName.endswith("*"):
          codeName=codeName.replace("/*", "")
          for putativeKey in h5File[codeName]:
            codeStr=h5File[codeName+'/'+putativeKey][0]
            exec(codeStr, env)  # normFun is in the code
            codes[putativeKey] =(env.get(putativeKey))
        else:
          codeStr =  h5File[codeName][0]
          exec(codeStr, env)  # normFun is in the code
          codes[os.path.basename(codeName)]= (env.get(os.path.basename(codeName)))
    return args, codes
  except KeyError:
    print("Error loading config from hd5")
    return None, None

def loadNormalizationFunFromModel(kerasCheckpointFname, binary_mask=None, noise_stats=None):
  args, codes = retrieveParamsFromHd5(kerasCheckpointFname, [], ['code/normFun' ])
  if codes is None:
    return None
  if binary_mask is not None:
    binary_mask, __ = loadVolIfFnameOrIgnoreIfMatrix(binary_mask, normalize=None)
    normalizationFunction = lambda x: codes['normFun'](x, binary_mask)
  else:
    normalizationFunction = codes['normFun']

  if noise_stats is not None:
    assert normalizationFunction.__name__=="inputNormalization_3"
    return lambda y: normalizationFunction(y, noise_stats)

  return  normalizationFunction


def loadChunkConfigFromModel(kerasCheckpointFname):
  args, codes = retrieveParamsFromHd5(kerasCheckpointFname, ['configParams/NNET_INPUT_STRIDE', 'configParams/NNET_INPUT_SIZE'], [])
  if args is None:
    return None
  else:
    return args

if __name__ == "__main__":
  from config import DEFAULT_MODEL_DIR
  print( DEFAULT_MODEL_DIR )
  fname = os.path.join(DEFAULT_MODEL_DIR, "deepEMhancer_highRes.hd5")
  conf=loadChunkConfigFromModel( fname); print(conf)
  conf = retrieveParamsFromHd5(fname, [], ["code/*"]); print(conf)
  #conf=loadChunkConfigFromModel("/home/ruben/Tesis/cryoEM_cosas/auto3dMask/data/nnetResults/vt2_m_28_checkpoints/bestCheckpoint_locscale_masked.hd5"); print(conf)
