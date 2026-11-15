import json, math, random, re, collections
D='/home/claude/base/toollinkos/toollinkos/'
tools={t['name']:t for t in json.load(open(D+'core_tools.json'))+json.load(open(D+'regular_tools.json'))}
inst=json.load(open(D+'instances.json'))
tok=lambda s: [w for w in re.findall(r'[a-z0-9]+', s.lower()) if w not in STOP]
STOP=set('a an the to of for and or in on my me i is it be can you please with from this that what at by as are do your'.split())
def tdoc(t): return tok(t['name'].replace('_',' ')+' '+t['description'])
docs={n:tdoc(t) for n,t in tools.items()}
N=len(docs); df=collections.Counter(w for d in docs.values() for w in set(d)); avg=sum(map(len,docs.values()))/N
def bm25(q,k=5):
  qs=tok(q); sc={}
  for n,d in docs.items():
    tf=collections.Counter(d); s=0
    for w in qs:
      if w in tf: idf=math.log(1+(N-df[w]+.5)/(df[w]+.5)); s+=idf*tf[w]*2.2/(tf[w]+1.2*(.25+.75*len(d)/avg))
    if s: sc[n]=s
  return sorted(sc,key=sc.get,reverse=True)[:k]
deps={n:[d['name'] for d in t.get('depends_on',[]) if d['name'] in tools] for n,t in tools.items()}
def closure(seeds,hops=3):
  out=list(seeds); fr=list(seeds)
  for _ in range(hops):
    nx=[]
    for s in fr:
      for d in deps[s]:
        if d not in out: out.append(d); nx.append(d)
    fr=nx
  return out
sizes=[len(i['golden_function_names']) for i in inst]; print('golden size avg',sum(sizes)/len(sizes),'max',max(sizes))
def score(fn,data):
  full=rec=ntool=0
  for i in data:
    g=set(i['golden_function_names']); got=fn(i['user_query']); ntool+=len(got)
    rec+=len(g&set(got))/len(g); full+= g<=set(got)
  n=len(data); return round(rec/n,3), round(full/n,3), round(ntool/n,1)
random.seed(7); random.shuffle(inst); test=inst[:300]; stream=inst[300:]
B=12
print('bm25 top12', score(lambda q: bm25(q,B),test))
print('bm25 top3 + deps cap12', score(lambda q: closure(bm25(q,3))[:B],test))
# learning: term->tool weights, tool co-use
T2=collections.defaultdict(collections.Counter); CO=collections.defaultdict(collections.Counter)
def learned(q):
  qs=set(tok(q)); sc=collections.Counter()
  for w in qs:
    tot=sum(T2[w].values())
    if not tot: continue
    idf=math.log(1+N/(df.get(w,0)+1))
    for t,c in T2[w].items(): sc[t]+=idf*c/tot
  return [t for t,_ in sc.most_common(3)]
def ours(q):
  seeds=[]
  for t in learned(q)+bm25(q,3):
    if t not in seeds: seeds.append(t)
  seeds=seeds[:4]
  out=closure(seeds)
  for s in seeds:
    for t,c in CO[s].most_common(3):
      if c>=2 and t not in out: out.append(t)
  return out[:B]
print('round 0', score(ours,test))
for r in range(1,13):
  for i in stream[(r-1)*100:r*100]:
    g=i['golden_function_names']
    for w in set(tok(i['user_query'])):
      for t in g: T2[w][t]+=1
    for a in g:
      for b in g:
        if a!=b: CO[a][b]+=1
  print('round',r,score(ours,test))
miss_main=miss_dep=over=0; miss_names=collections.Counter()
for i in test:
  g=set(i['golden_function_names']); got=ours(i['user_query'])
  if g<=set(got): continue
  if i['main_golden_function_name'] not in got: miss_main+=1
  else:
    full=closure([i['main_golden_function_name']],5)
    if g<=set(full) and not g<=set(got): over+=1
    else: miss_dep+=1; miss_names.update(g-set(full))
print('fail: main missing',miss_main,'cap overflow',over,'golden not in dep closure',miss_dep)
print(miss_names.most_common(10))
