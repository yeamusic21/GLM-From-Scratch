# D = 2 * sum[
# y * log(y / prediction)
# - (y - prediction)
import numpy as np

y = 3
prediction = 4
print("====================================")
print(f"y is {y} and prediction is {prediction}")
print("====================================")

D_part_1_0 = np.log(y / prediction) 
D_part_1_1 = y * D_part_1_0
D_part_1_2 = (y - prediction)
print(f"log({y} / {prediction}) --> ", D_part_1_0)
print(f"{y} * {D_part_1_0}) --> ", D_part_1_1)
print(f"({y} - {prediction}) --> ", D_part_1_2)

D_part_1 = D_part_1_1 - D_part_1_2
print(f"{D_part_1_1} - {D_part_1_2} --> ", D_part_1)

