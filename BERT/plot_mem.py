import re
import matplotlib.pyplot as plt

# epochs = []
# losses = []

# with open('output_557.txt', 'r') as f:
#     for line in f:
#         # 提取 Last_epoch
#         match_epoch = re.search(r'Last_epoch:(\d+)', line)
#         if match_epoch:
#             epoch = int(match_epoch.group(1))
#
#         # 提取 Loss
#         match_loss = re.search(r'Training/Loss ([\d\.]+)', line)
#         if match_loss:
#             loss = float(match_loss.group(1))
#             # 这里默认每条loss对应一个epoch
#             epochs.append(epoch)
#             losses.append(loss)
#
# epochs_base = []
# losses_base = []

# with open('output_base565.txt', 'r') as f:
#     for line in f:
#         # 提取 Last_epoch
#         match_epoch = re.search(r'Last_epoch:(\d+)', line)
#         if match_epoch:
#             epoch = int(match_epoch.group(1))
#
#         # 提取 Loss
#         match_loss = re.search(r'Training/Loss ([\d\.]+)', line)
#         if match_loss:
#             loss = float(match_loss.group(1))
#             # 这里默认每条loss对应一个epoch
#             epochs_base.append(epoch)
#             losses_base.append(loss)
#
#
# epochs_1 = []
# losses_1 = []
#
# with open('output_582.txt', 'r') as f:
#     for line in f:
#         # 提取 Last_epoch
#         match_epoch = re.search(r'Last_epoch:(\d+)', line)
#         if match_epoch:
#             epoch = int(match_epoch.group(1))
#
#         # 提取 Loss
#         match_loss = re.search(r'Training/Loss ([\d\.]+)', line)
#         if match_loss:
#             loss = float(match_loss.group(1))
#             # 这里默认每条loss对应一个epoch
#             epochs_1.append(epoch)
#             losses_1.append(loss)
#
# epochs_2 = []
# losses_2 = []
# with open('output_585.txt', 'r') as f:
#     for line in f:
#         # 提取 Last_epoch
#         match_epoch = re.search(r'Last_epoch:(\d+)', line)
#         if match_epoch:
#             epoch = int(match_epoch.group(1))
#
#         # 提取 Loss
#         match_loss = re.search(r'Training/Loss ([\d\.]+)', line)
#         if match_loss:
#             loss = float(match_loss.group(1))
#             # 这里默认每条loss对应一个epoch
#             epochs_2.append(epoch)
#             losses_2.append(loss)
#
#
# epochs_2 = []
# losses_2 = []
# with open('output_585.txt', 'r') as f:
#     for line in f:
#         # 提取 Last_epoch
#         match_epoch = re.search(r'Last_epoch:(\d+)', line)
#         if match_epoch:
#             epoch = int(match_epoch.group(1))
#
#         # 提取 Loss
#         match_loss = re.search(r'Training/Loss ([\d\.]+)', line)
#         if match_loss:
#             loss = float(match_loss.group(1))
#             # 这里默认每条loss对应一个epoch
#             epochs_2.append(epoch)
#             losses_2.append(loss)
#
# epochs_3 = []
# losses_3 = []
# with open('output_589_base_LN.txt', 'r') as f:
#     for line in f:
#         # 提取 Last_epoch
#         match_epoch = re.search(r'Last_epoch:(\d+)', line)
#         if match_epoch:
#             epoch = int(match_epoch.group(1))
#
#         # 提取 Loss
#         match_loss = re.search(r'Training/Loss ([\d\.]+)', line)
#         if match_loss:
#             loss = float(match_loss.group(1))
#             # 这里默认每条loss对应一个epoch
#             epochs_3.append(epoch)
#             losses_3.append(loss)
#
# epochs_4 = []
# losses_4 = []
# with open('output_620_TTFS.txt', 'r') as f:
# # with open('output_586_TTFS.txt', 'r') as f:
#     for line in f:
#         # 提取 Last_epoch
#         match_epoch = re.search(r'Last_epoch:(\d+)', line)
#         if match_epoch:
#             epoch = int(match_epoch.group(1))
#
#         # 提取 Loss
#         match_loss = re.search(r'Training/Loss ([\d\.]+)', line)
#         if match_loss:
#             loss = float(match_loss.group(1))
#             # 这里默认每条loss对应一个epoch
#             epochs_4.append(epoch)
#             losses_4.append(loss)
#
# epochs_5 = []
# losses_5 = []
# with open('output_590_base_noLN.txt', 'r') as f:
#     for line in f:
#         # 提取 Last_epoch
#         match_epoch = re.search(r'Last_epoch:(\d+)', line)
#         if match_epoch:
#             epoch = int(match_epoch.group(1))
#
#         # 提取 Loss
#         match_loss = re.search(r'Training/Loss ([\d\.]+)', line)
#         if match_loss:
#             loss = float(match_loss.group(1))
#             # 这里默认每条loss对应一个epoch
#             epochs_5.append(epoch)
#             losses_5.append(loss)

epochs_6 = []
losses_6 = []
with open('mem.txt', 'r') as f:
    for line in f:
        # 提取 Last_epoch
        match = re.match(r"\[(\d+)\] Mem Used: ([\d.]+)GB", line)
        if match:
            step = int(match.group(1))
            mem = float(match.group(2))


            # 这里默认每条loss对应一个epoch
            # epochs_6.append(step)
            losses_6.append(mem)

epochs_6 = list(range(len(losses_6)))
plt.figure(figsize=(8, 5))
# plt.plot(epochs, losses, marker='o', color="green", label='bf16-TTFS')
# plt.plot(epochs_base, losses_base, marker='x', color="orange", label='f32_TTFS')
# plt.plot(epochs_1, losses_1, marker='.', color="blue", label='TTFS_new_LN1')
# plt.plot(epochs_2, losses_2, marker='*', color="red", label='TTFS-withoutembeddingLNlayer')
# plt.plot(epochs_3, losses_3, marker='_', color="pink", label='Base+LNlayerinoutput')
# plt.plot(epochs_4, losses_4, marker='.', color="grey", label='TTFS_new_LN2')
# plt.plot(epochs_5, losses_5, marker='_', color="green", label='Base+output')
plt.plot(epochs_6, losses_6, marker='_', color="yellow", label='mem')

plt.xlabel('step')

plt.ylabel('Mem')
plt.title('Mem')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("1")
