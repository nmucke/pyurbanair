
import xarray
import pdb
import matplotlib.pyplot as plt

state = xarray.open_dataset("state.nc")
ss = state.isel(time=0)
ss = state.isel(zt=1)
ss = state.u.values

pdb.set_trace()


plt.figure()
plt.imshow(ss.u.values[: :, 0])
plt.savefig(lol.png)
plt.show()





