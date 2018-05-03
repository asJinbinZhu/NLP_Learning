list = ['physics', 'chemistry', 1989, 2018, 15.5, 'heihei', 90]

# 1 - visit list item
print('the first: ', list[0])
print('the last: ', list[-1])
print('the 1-5th: ', list[1: 5]) # expect the 5th
print('the first 5th: ', list[: 5])
print('all after the 2th: ', list[1: ])

# 2 - update the list
# add
list.append('zhangsan')
print(list)

# delete
del list[0]
print(list)

# 3 - others
print('length of list: ', len(list))


