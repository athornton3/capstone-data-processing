# Clean, transform, and encode given data in csv
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer, util
import pickle
import time
import re
import urllib.parse
import s3fs
import pgeocode

#function to take title & abstract and search corpus for similar projects
def search_projects(title, abstract):
    query_embedding = model.encode(title+'[SEP]'+abstract, convert_to_tensor=True)

    results = util.semantic_search(query_embedding, corpus_embeddings, top_k = 10)
    #results_normalized = util.semantic_search(query_embedding, corpus_embeddings_norm, score_function=util.dot_score, top_k = 10)
    
    print("\n\nProject:", title)
    print("Most similar projects:")
    #for prj in results[0]:
    #    related_project = projects_df.loc[prj['corpus_id']]
        #print("{:.3f}\t{}\n\t{} {} {}".format(prj['score'], related_project['AwardTitle'], related_project['Investigator-PI_FULL_NAME'], related_project['ProgramReference-Text-1'], related_project['Institution-Name']))   
    
    #for prj in results_normalized[0]:
    #    related_project = projects_df.loc[prj['corpus_id']]
        #print("{:.3f}\t{}\n\t{} {} {}".format(prj['score'], related_project['AwardTitle'], related_project['Investigator-PI_FULL_NAME'], related_project['ProgramReference-Text-1'], related_project['Institution-Name']))   
    df = pd.DataFrame()
    scores = []
    
    #make sure projects_df is re-read after all the data cleaning or indeces get weird
    for prj in results[0]:
        related_project = projects_df.loc[prj['corpus_id']]
        scores.append(prj['score'])
        df = df.append(related_project)
    df.insert(0, "cosim_score", scores)
    return df

# ----main----
fs = s3fs.S3FileSystem(anon=False)
#create specter model
model = SentenceTransformer('allenai-specter')

#read in data
#projects_df = pd.read_csv("export_21_22_23_col_removed.csv", encoding='utf-8')
#projects_df = pd.read_csv("export_21_22_23_col_rv_100_latlong.csv", encoding='utf-8')
projects_df = pd.read_csv("export_2023_col_rv_latlong_narrow_topics.csv", encoding='utf-8')

# take a sample 
#projects_df.sample(frac = 0.01) #take a percentage of records
#projects_df.sample(n = 100)

#data cleaning for abstracts
#---remove duplicates on title and abstract
duplicateRows = projects_df[projects_df.duplicated(['AwardTitle', 'AbstractNarration'])]
projects_df.drop_duplicates(['AwardTitle', 'AbstractNarration'], inplace =True)
#print(duplicateRows) #42 dups in 2023

#---strip out gt lt &lt;br/&gt;
projects_df['AbstractNarration']=projects_df['AbstractNarration'].str.replace("br/"," ")
projects_df['AbstractNarration']=projects_df['AbstractNarration'].str.replace("&lt;"," ")
projects_df['AbstractNarration']=projects_df['AbstractNarration'].str.replace("&gt;"," ")

projects_df['AbstractNarration'] = projects_df['AbstractNarration'].map(lambda x: re.sub('[^A-Za-z0-9 ?().,-]+','', str(x)))

#data transformations
#---limit size to 100 words in each title and 300 words in each abstract to keep encodings size down
max_size = 100
projects_df['AwardTitle'] = projects_df['AwardTitle'].apply(lambda x: ' '.join(x.split(maxsplit=max_size)[:max_size]))                                                                       

max_size = 300
projects_df['AbstractNarration'] = projects_df['AbstractNarration'].apply(lambda x: ' '.join(x.split(maxsplit=max_size)[:max_size]))  

#---add institutions latitude and longitude according to zipcode
nomi = pgeocode.Nominatim("us")
projects_df['Institution-ZipCode'] = projects_df['Institution-ZipCode'].astype(str).str.slice(0, 5)
zipcodes = projects_df['Institution-ZipCode'].apply(lambda x:nomi.query_postal_code(x))
projects_df = pd.concat([projects_df, zipcodes], axis="columns")
projects_df.to_csv("export_2023_col_rv_latlong.csv", encoding='utf-8')

#create data list to feed to model
project_texts = projects_df['AwardTitle'].astype(str) + '[SEP]' + projects_df['AbstractNarration'].astype(str)

sentences_array = project_texts.to_numpy()
                                                   
#do the work -- takes a long time to run
start_time = time.time()
corpus_embeddings = model.encode(project_texts, convert_to_tensor=True)
print("--- %s seconds ---" % (time.time() - start_time))
#save to s3 buckeet
pickle.dump(corpus_embeddings, fs.open(f"streamlitbucketcapstoneajt/corpus_embeddings_2023.pkl",'wb'))

#or read saved encodings from buckeet
#with fs.open("streamlitbucketcapstoneajt/corpus_embeddings_2023.pkl", 'rb') as pkl:
#    cache_data = pickle.loads(pkl.read())
#corpus_embeddings = cache_data

# normalize embeddings to 1 for speed optimization
start_time = time.time()
corpus_embeddings_norm = corpus_embeddings.to('cpu')
corpus_embeddings_norm = util.normalize_embeddings(corpus_embeddings_norm)
print("--- %s seconds ---" % (time.time() - start_time))

# run the function 
start_time = time.time()
df = pd.DataFrame()
df = search_projects(title='Specializing Word Embeddings (for Parsing) by Information Bottleneck',
              abstract='Pre-trained word embeddings like ELMo and BERT contain rich syntactic and semantic information, resulting in state-of-the-art performance on various tasks. We propose a very fast variational information bottleneck (VIB) method to nonlinearly compress these embeddings, keeping only the information that helps a discriminative parser. We compress each word embedding to either a discrete tag or a continuous vector. In the discrete version, our automatically compressed tags form an alternative tag set: we show experimentally that our tags capture most of the information in traditional POS tag annotations, but our tag sequences can be parsed more accurately at the same level of tag granularity. In the continuous version, we show experimentally that moderately compressing the word embeddings by our method yields a more accurate parser in 8 of 9 languages, unlike simple dimensionality reduction.')
print("--- %s seconds ---" % (time.time() - start_time))

#function to take title & abstract and search corpus for similar projects, compare normalized with non-normalized embeddings
def search_projects_compare(title, abstract):
    
    start_time = time.time()
    query_embedding = model.encode(title+'[SEP]'+abstract, convert_to_tensor=True)
    print("Encode submitted award --- %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    results = util.semantic_search(query_embedding, corpus_embeddings, top_k = 10)
    print("Match with cosine --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    results_dotproduct = util.semantic_search(query_embedding, corpus_embeddings, score_function=util.dot_score, top_k = 10)   
    print("Match with dot product --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    results_normalized = util.semantic_search(query_embedding, corpus_embeddings_norm, score_function=util.dot_score, top_k = 10)
    print("Match with normalized dot product --- %s seconds ---" % (time.time() - start_time))
    
    print("\n\nProject:", title)
    print("Matching projects - Cosine Similarity:")
    
    for prj in results[0]:
        related_project = projects_df.loc[prj['corpus_id']]
        print("{:.3f}\t{}".format(prj['score'], related_project['AwardTitle'][:20]+"..."))   
    
    print("\n\nProject:", title)
    print("Most similar project - Dot Product:")
    for prj in results_dotproduct[0]:
        related_project = projects_df.loc[prj['corpus_id']]
        print("{:.3f}\t{}".format(prj['score'], related_project['AwardTitle'][:20]+"..."))   
    
    print("\n\nProject:", title)
    print("Most similar project - Normalized Dot Product:")
    for prj in results_normalized[0]:
        related_project = projects_df.loc[prj['corpus_id']]
        print("{:.3f}\t{}".format(prj['score'], related_project['AwardTitle'][:20]+"..."))   

    return df
  
# run the comparison function 
start_time = time.time()
df = pd.DataFrame()
df = search_projects_compare(title='CAREER: Models and Algorithms for Strategic Conservation Planning',
              abstract='This Faculty Early Career Development Program (CAREER) award will contribute to the national health, prosperity, and welfare by improving decision-making in conservation planning through new theory, models, algorithms, and visual analytics tools for landscape and conservation ecology. Biodiversity has been declining at rapid rates during the last several decades due to habitat loss, landscape deterioration, environmental change, and human-related activities that directly and indirectly affect natural habitats. In addition to its economic and cultural value, biodiversity plays an important role in keeping an environments ecosystem in balance. Disrupting such processes can reduce the provision of natural resources such as food and water, which in turn yields a direct threat to human health. Protecting natural areas is fundamental to preserving biodiversity and to mitigate the effects of ongoing environmental change. This award will contribute quantitative methods to support informed decisions on conservation design and effective land use to support species sustainability. These methods integrate realistic ecological features, specific spatial properties of the selected reserves (e.g., connectivity), population dynamics within the spatial assets, and the impact of current and future threats. The educational plan will improve the skills and diversity of future generations of engineers via technical training and engagement in transdisciplinary research. The outreach activities aim to increase the students awareness of current biodiversity and conservation challenges. This award supports fundamental research on the design of portfolios of land or marine patches to support species sustainability. These design problems result in very large mixed-integer linear programs whose solutions require innovative formulations and new large-scale optimization methods. The new models and specialized algorithms will allow decision-makers to solve a variety of realistic large-scale corridor and reserve design problems that include patch-specific conservation decisions under spatial, operational, ecological, and biological requirements. ')
print("--- %s seconds ---" % (time.time() - start_time))

#paraphrase mining - many-to-many
start_time = time.time()
paraphrases = util.paraphrase_mining(model, project_texts)
print("--- %s seconds ---" % (time.time() - start_time))

#look at the top 100 matches, remove 1.0000 matches
for paraphrase in paraphrases[0:100]:
    score, i, j = paraphrase
    if score < 1 and (project_texts[i][:75] != project_texts[j][:75]):
        print("Score: {:.4f}\t{}\t{}\t ".format(score,project_texts[i], project_texts[j]))
