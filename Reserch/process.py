with open("ng.txt", "r", encoding="UTF-8") as f:
    l_ng = f.readlines()
f.close()

with open("ok.txt", "r", encoding="UTF-8") as f:
    l_ok = f.readlines()
f.close()

sorted(l_ng)
sorted(l_ok)

print(l_ng[-1] < l_ok[0])

####
with open(
    "Dataset/Crash/Crash-1500/Frames/video_label.txt", "w", encoding="UTF-8"
) as f:
    for i in range(500):
        f.write(f"{i}:0\n")
    f.close()
