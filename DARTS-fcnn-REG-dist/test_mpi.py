import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

# Verify that MPI multi-threading is supported
print(hvd.mpi_threads_supported())
assert hvd.mpi_threads_supported()

from mpi4py import MPI

print(hvd.size())
assert hvd.size() == MPI.COMM_WORLD.Get_size()
