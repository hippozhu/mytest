#Python 3 for gateway

import requests as req
import csv
import sys
import jwt
import time
import os
import glob
import xml.etree.ElementTree as ET
import queue
import threading
import json
import pickle
import gzip

def get_token(threadname):
    data = {"grant_type": "client_credentials", "client_assertion_type":
        "urn:ietf:params:oauth:client-assertion-type:jwt-bearer", "client_assertion":
        jwt.encode({"iss": "4190c1c3", "sub": "4190c1c3", "aud":
            "http://www.careerbuilder.com/share/oauth2/token.aspx", "exp": time.time() + 900},
            "mXL4HdeEGvfCxthZJpD88sXTmxmGWi39Xu6oOYYyAeDyvfCbqRRMkPBkKqVzcMkDHJ1ji7N6bZToepba8mx8YA==",
            "HS512"), "client_id": "4190c1c3"}
    urlpro = "https://www.careerbuilder.com/share/oauth2/token.aspx"
    r = req.post(urlpro, data=data)
    if r.status_code == 200:
        r2 = r.json()
        token = 'bearer ' + str(r2['access_token'])
        print('new token generated: ' + threadname + " - " + token)
        return token
    else:
        print(threadname + ' ' + r.text)
        time.sleep(5)
        return get_token(threadname)


def get_skills(token, description, version, threadname, jobdid):
    url = "https://api.careerbuilder.com/core/tagging/skills"
    headers = {"Authorization": token}
    data = {"version": version}
    data["content"] = description
    request = req.post(url, data=data, headers=headers)
    if request.status_code == 200:
        reqjson = request.json()

        return reqjson
    else:
        print("%s Error code:%s, %s" %(jobdid, str(request.status_code), str(request.text)))
        #print(threadname + " - " + str(token))
        return None

def xml_job_iterator():
    #xml_job_files = glob.iglob('data_test/*.xml')
    xml_job_files = glob.iglob('data/*.xml')
    t0 = time.time()
    for i, job_file in enumerate(xml_job_files):
        if (i+1) % 2000 == 0:
            print('read %d in %.1fs' % (i+1, time.time()-t0))
        root = ET.parse(job_file).getroot()
        yield dict([('jobdid', root.find('DID').text),
            ('title', root.find('JobTitle').text),
            ('description', root.find('JobDescription').text)])

def csv_job_iterator():
    with open('jobs2.2m.csv') as inputfile:
        reader = csv.DictReader(inputfile, fieldnames=['jobdid', 'title', 'description'])
        t0 = time.time()
        for i, job in enumerate(reader):
            if (i+1) % 500000 == 0:
                print('read %d in %.1fs' % (i+1, time.time()-t0))
            yield job
        print('Done reading')

def job_iterator():
    job_did = set([line.strip() for line in open('jobs1.1m_DID')])
    with gzip.open('jobs5m.csv.gz', 'rt') as inputfile:
        reader = csv.DictReader(inputfile, fieldnames=['jobdid', 'title', 'description', 'req'])
        t0 = time.time()
        for i, job in enumerate(reader):
            if (i+1) % 100000 == 0:
                print('read %d in %.1fs' % (i+1, time.time()-t0))
            if job['jobdid'] in job_did and len(job['description'])>0:
                yield job
            else:
                continue
        print('Done reading')

def job_iterator_1500():
    with open('jobs1500_deduplicate.csv', encoding="ISO-8859-1") as inputfile:
        reader = csv.DictReader(inputfile, fieldnames=['jobdid', 'title', 'description'])
        next(reader)
        t0 = time.time()
        for i, job in enumerate(reader):
            if (i+1) % 5000 == 0:
                print('read %d in %.1fs' % (i+1, time.time()-t0))
            yield job
        print('Done reading')

def get_skills_yun(inq, outq):
    while True:
        job = inq.get()
        try:
            result = get_skills(token, job['description'], version, threadname, job['jobdid'])
            if result is None:
                continue
            skills = list(map(lambda r: r['normalized_term'], result['data']))
            #print(skills)
            outq.put((job['jobdid'], skills))
            inq.task_done()
        except TypeError:
            print('TypeError', job['jobdid'], result)

def output_skills_yun(outq, csvwriter):
    global threadname
    global token
    t0 = time.time()
    count = 0
    while True:
        item = outq.get()
        csvwriter.writerow([item[0], json.dumps(item[1])])
        outq.task_done()
        count += 1
        if count % 2000 == 0:
            print('write %d in %.1fs' % (count, time.time()-t0))
        if count % 100000 == 0:
            threadname += '1'
            token = get_token(threadname)

def main_yun():
    #iter_job = iter(xml_job_iterator())
    #iter_job = iter(csv_job_iterator())
    iter_job = iter(job_iterator())
    #iter_job = iter(job_iterator_1500())
    inq = queue.Queue()
    outq = queue.Queue()
    threads = []
    for _ in range(10):
        t = threading.Thread(target=get_skills_yun, args=(inq, outq))
        t.daemon = True
        t.start()
        threads.append(t)

    #rest_did = pickle.load(open('rest.pickle', 'rb'))
    with open('Skillsv4/output.csv', 'w') as fout:
        csvwriter = csv.writer(fout)
        t = threading.Thread(target=output_skills_yun, args=(outq, csvwriter))
        t.daemon = True
        t.start()
        #t.join()

        for i, job in enumerate(iter_job):
            '''
            if i<426000:
                continue
            if job['jobdid'] in rest_did:
                inq.put(job)
            '''
            inq.put(job)
        inq.join()
        outq.join()

    #for t in threads:
    #    t.join()

version = '4.1'
csv.field_size_limit(sys.maxsize)
threadname = str(threading.current_thread().name)
token = get_token(threadname)

if __name__ == '__main__':
    main_yun()
