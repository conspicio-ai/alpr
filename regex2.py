from re import compile
import re
plate_format = '[A-Z]{2}.*[0-9]{2}.*[A-Z]{0,3}.*[0-9]{4}$'
#replace_format = '[A-Z]{2}[0-9]{2}[A-Z]{0,3}[0-9]{4}$'
# delhi_format = compile('^[A-Z]{2}\s[0-9]{1}[A-Z]{1}\s[A-Z]{2}\s[0-9]{4}$')

text = "MH 31 EU 9949"


# if plate_format.match(text) is not None:
# 	print('ok')
# else:
# 	print('not ok')
x=re.findall(plate_format,text,re.MULTILINE)

x=''.join(x)

#print(x)

#x = re.Replace(x, @"[^0-9a-zA-Z]+", "")
x = re.sub('[^A-Za-z0-9]+', '', x)
y= re.sub('\s*\'*[a-z]*','',x)

"""def repl(str):
	n1=re.search('[0-9]{2}',x)
	n1x,n1y=n1.span()
	n2=re.search('.[A-Z]{2}',x)
	n2x,n2y=n2.span()
	##n3=re.search('[0-9]{4}',x)
	return str[0:2]+str[n1x:n1y] + str[n2x+1:n2y] + str[-4::]"""



print(y)
