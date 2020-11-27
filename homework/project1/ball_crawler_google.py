from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib.request
import time

#찾고자 하는 검색어를 url로 만들어 준다.
searchterm = 'soccer ball'
url = "https://www.google.com/search?q="+searchterm+"&source=lnms&tbm=isch"
path = './data/project1/ball'
# chrom webdriver 사용하여 브라우저를 가져온다.
browser = webdriver.Chrome('./homework/project1/chromedriver.exe')
browser.get(url)


header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
# 이미지 카운트 (이미지 저장할 때 사용하기 위해서)
counter = 651
succounter = 0

# l = browser.find_elements_by_class_name('rg_i')
# print(len(l))


time.sleep(0.5)
for i in range(50):
    
    if i %10 == 0:
        print(i)
    time.sleep(0.5)
    browser.execute_script('window.scrollBy(0,10000)')




for x in browser.find_elements_by_class_name('rg_i'):
    counter = counter + 1
    x.screenshot(str(counter)+'.png')

    succounter = succounter + 1
            
print (succounter, "succesfully downloaded")
browser.close()