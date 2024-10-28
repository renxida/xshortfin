import shortfin as sf
import shortfin.host
import shortfin.array as sfnp
import shortfin.amdgpu

gpu = True
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

dtype = sfnp.float16

value = 3

ary = sfnp.device_array.for_host(fiber.device(0), [2, 4], dtype)
with ary.map(discard=True) as m:
    m.fill(value)



a = sfnp.device_array.for_device(fiber.device(0), [2, 4], dtype)
with a.map(discard=True) as m:
    m.fill(0.0)

def to_host(arr: sfnp.device_array):
    harr = arr.for_transfer()
    harr.copy_from(arr)
    return harr
b = to_host(a)

hary = ary.items
print(hary)

print(ary.__repr__())

# readback = ary.items.tolist()
# assert readback == [value] * 8