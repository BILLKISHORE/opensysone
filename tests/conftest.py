import mlx.core as mx

if not mx.metal.is_available():
    mx.set_default_device(mx.cpu)
