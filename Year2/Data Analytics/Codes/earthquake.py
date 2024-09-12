richter_scale = float(input("Enter a value for Richter scale result:"))

if 1 < richter_scale < 2:
    print("Microearthquakes are not felt!!!")
elif 2 <= richter_scale < 4:
    print("Very rarely causes damage!")
elif 4 <= richter_scale < 5:
    print("Damage done to weak buildings!")
elif 5 <= richter_scale < 6:
    print("Causes damage to poorly constructed buildings!!!")
elif 6 <= richter_scale < 7:
    print("Causes damage to well-built structures!!")
elif 7 <= richter_scale < 8:
    print("Causes damage to most buildings!!")
elif 8 <= richter_scale < 9:
    print("Causes major destruction!!")
else:
    print("Causes unbelievable damage!!!")
