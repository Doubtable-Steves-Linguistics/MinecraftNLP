import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import time

'''
This script will scrape for up to 5000 urls for minecraft related repos from github.
It will save those urls as a csv file in the folder the script is located in.
It requires around 100 minutes to run.
'''

#df = pd.DataFrame(columns= ['repo_urls'])
def get_minecraft_repositories():

    # creates list, urls to fill with hyperlinks gathered from github
    urls = []
    
    # key loop, begins with hitting github server and pulling the information for a specifiied page
    for i in range(1, 501):
        '''
        in the search portion of github, specifies the page number 'i', searching for 
        minecraft associated repositories
        '''
        try:
            url = f'https://github.com/search?p={i}&q=minecraft&type=Repositories'

            # gets webpage information to parse for data
            reqs = requests.get(url)

            # creates a text file which has been parsed, expecting the file to be using html code
            soup = BeautifulSoup(reqs.text, 'html.parser')

            # pauses 5 seconds before moving on to the next task
            time.sleep(5)
            
            # looks through html-parsed file, looking for hyperlink references, appending them to urls list
            for link in soup.find_all('a',class_="v-align-middle"):
                hyperlink = re.sub(r'/','', link.get('href'), count = 1)
                urls.append(hyperlink)
                #df.loc[len(hyperlink)] = hyperlink[link]
                time.sleep(0.5)
        
        except:
            
            pass 
    
    # takes the urls list and forms it into a dataframe
    df = pd.DataFrame(columns = ['repo_link'])
    df['repo_link'] = urls
    
    # saves dataframe to disk
    df.to_csv('large_url_csv.csv', index=False)
    
    
    
get_minecraft_repositories()