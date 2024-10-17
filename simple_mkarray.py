import shortfin as sf
import shortfin.host
import shortfin.array as sfnp
import shortfin.amdgpu

gpu = False
if gpu:
    sc = sf.amdgpu.SystemBuilder()
else:
    sc = sf.host.CPUSystemBuilder()
lsys = sc.create_system()



def fiber(lsys):
    return lsys.create_fiber()


def device(fiber):
    return fiber.device(0)

fiber = fiber(lsys)

device = device(fiber)

dtype = sfnp.float32

value = 3

ary = sfnp.device_array.for_host(fiber.device(0), [2, 4], dtype)
with ary.map(discard=True) as m:
    m.fill(value)

print(ary.__repr__())

# readback = ary.items.tolist()
# assert readback == [value] * 8