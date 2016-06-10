import glob
import xml.etree.ElementTree as ET
import time
import string
import re
import csv
import sys
import gzip 
import zipfile
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize

regex_punc = re.compile('[%s]' % re.escape(string.punctuation))
stemmer = PorterStemmer()

def xml_job_iterator_dict(job_file_folder):
    xml_job_files = glob.iglob(job_file_folder + '/*.xml')
    t0 = time.time()
    for i, job_file in enumerate(xml_job_files):
        if (i+1) % 1000 == 0:
            print('read %d in %.1fs' % (i+1, time.time()-t0))
        root = ET.parse(job_file).getroot()
        yield dict([('jobdid', root.find('DID').text),
            ('title', root.find('JobTitle').text),
            ('description', root.find('JobDescription').text)])

def xml_job_iterator(job_file_folder):
    xml_job_files = glob.iglob(job_file_folder + '/*.xml')
    t0 = time.time()
    for i, job_file in enumerate(xml_job_files):
        if (i+1) % 10000 == 0:
            print('read %d in %.1fs' % (i+1, time.time()-t0))
        root = ET.parse(job_file).getroot()
        yield (root.find('DID').text, root.find('JobTitle').text,
            root.find('JobDescription').text)

def job_corpus(job_file_folder):
    xml_job_files = glob.iglob(job_file_folder + '/*.xml')

def my_tokenizer(text):
  return list(map(stemmer.stem, word_tokenize(regex_punc.sub(' ', text))))

def clean_escape_char_nolower(text):
  if text == None:
    return ''
  text = ''.join(i if i in string.printable else ' ' for i in text)
  text = re.sub('&amp;', u'&', re.sub('&gt;', u'>', re.sub('&lt;', u'<', text)))
  text = re.sub('<[^>]*>', u' ', text)
  text = re.sub('\n|&\w+;|&#\w+;', u' ', text)
  return text

def clean_txt(text):
  return re.sub('\s+', ' ', re.sub(r'[^\x00-\x7F]+',' ', clean_escape_char_nolower(text).strip()))

def extract_job_data(zf):
    with open('jobs5m.csv', 'w') as fout:
        csvwriter = csv.writer(fout)
        for i, file_path in enumerate(zf.namelist()):
            file_name = file_path.split('/')[1]
            if file_name!='' and file_name[0]=='J':
                try:
                    if file_name[-1]=='z':
                        root = ET.parse(gzip.open(zf.open(file_path)))
                    else:
                        root = ET.parse(zf.open(file_path))
                    jobdid = file_name.split('.')[0]
                    title = root.find('JobTitle').text
                    desc = root.find('JobDescription').text
                    req = '' if root.find('JobRequirements') is None else root.find('JobRequirements').text
                    csvwriter.writerow([jobdid, clean_txt(title), clean_txt(desc), clean_txt(req)])
                except:
                    print(sys.exc_info()[0], file_path)
                    break
            if (i+1)%100000==0:
                print(i+1)

if __name__ == '__main__':
    zf = zipfile.ZipFile('data_5p7.zip', 'r')
    extract_job_data(zf)
