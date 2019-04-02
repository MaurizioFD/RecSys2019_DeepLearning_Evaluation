import numpy as np
import random
import time
#from time import time
#random.seed(123)
#ml 100k
usize = 943 + 1
msize = 1682 + 1
tsize = 18 + 1
#ml 1m
#usize = 6040 + 1
#msize = 3706 + 1

class MetapathBasePathSample:
    def __init__(self, **kargs):
        self.metapath = kargs.get('metapath')
        self.walk_num = kargs.get('walk_num')
        self.K = kargs.get('K')
        self.um_dict = dict()
        self.mu_dict = dict()
        self.mt_dict = dict()
        self.tm_dict = dict()
        self.uo_dict = dict()
        self.ou_dict = dict()
        self.ua_dict = dict()
        self.au_dict = dict()
        self.uu_dict = dict()
        self.mm_dict = dict()
        #self.um_list = list()
        
        self.user_embedding = np.zeros((usize, 64))
        self.item_embedding = np.zeros((msize, 64))
        self.type_embedding = np.zeros((tsize, 64))
        print('Begin to load data')
        start = time.time()

        self.load_user_embedding('../data/ml-100k.bpr.user_embedding')
        self.load_item_embedding('../data/ml-100k.bpr.item_embedding')
        self.load_type_embedding('../data/ml-100k.bpr.type_embedding')


        self.load_um(kargs.get('umfile'))
        self.load_mt(kargs.get('mtfile'))
        self.load_uu(kargs.get('uufile'))
        self.load_mm(kargs.get('mmfile'))
        #self.load_uo(kargs.get('uofile'))
        end = time.time()
        print('Load data finished, used time %.2fs' % (end - start))
        self.path_list = list()
        self.outfile = open(kargs.get('outfile_name'), 'w')
        self.metapath_based_randomwalk()
        self.outfile.close()

    def load_user_embedding(self, ufile):
        with open(ufile) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                for j in range(len(arr[1:])):
                    self.user_embedding[i][j] = float(arr[j + 1])

    def load_item_embedding(self, ifile):
        with open(ifile) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                for j in range(len(arr[1:])):
                    self.item_embedding[i][j] = float(arr[j + 1])
        
    def load_type_embedding(self, tfile):
        with open(tfile) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                for j in range(len(arr[1:])):
                    self.type_embedding[i][j] = float(arr[j + 1])
    def metapath_based_randomwalk(self):
        pair_list = []
        for u in range(1, usize):
            for i in range(1, msize):
                pair_list.append([u, i])
        print('load pairs finished num = ', len(pair_list))
        ctn = 0
        t1 = time.time()
        avg = 0
        for u, m in pair_list:
            ctn += 1
            #print u, m
            if ctn % 10000 == 0:
                print ('%d [%.4f]\n' % (ctn, time.time() - t1))
            if self.metapath == 'umum':
                path = self.walk_umum(u, m)
            elif self.metapath == 'umtm':
                path = self.walk_umtm(u, m)
            elif self.metapath == 'mumt':
                path = self.walk_mumt(u, m)
            elif self.metapath == 'mumumt':
                path = self.walk_mumumt(u, m)
            elif self.metapath == 'uuum':
                path = self.walk_uuum(u, m)
            elif self.metapath == 'ummm':
                path = self.walk_ummm(u, m)
            else:
                print('unknow metapath.')
                exit(0)
    
    def get_sim(self, u, v):
        return u.dot(v) / ((u.dot(u) ** 0.5) * (v.dot(v) ** 0.5))

    def walk_umum(self,s_u, e_m):
        limit = 10
        m_list = []
        for m in self.um_dict[s_u]:
            sim = self.get_sim(self.user_embedding[s_u], self.item_embedding[m])#self.user_embedding[s_u].dot(self.item_embedding[m]) / 
            m_list.append([m, sim])
        m_list.sort(key = lambda x:x[1], reverse = True)
        m_list = m_list[:min(limit, len(m_list))]
        
        u_list = []
        for u in self.mu_dict.get(e_m, []):
            sim = self.get_sim(self.item_embedding[e_m], self.user_embedding[u])#self.item_embedding[e_m].dot(self.user_embedding[u])
            u_list.append([u, sim])
        u_list.sort(key = lambda x:x[1], reverse = True)
        u_list = u_list[:min(limit, len(u_list))]

        mu_list = []
        for m in m_list:
            for u in u_list:
                mm = m[0]
                uu = u[0]
                if mm in self.mu_dict and uu in self.mu_dict[mm] and uu != s_u and mm != e_m:
                    sim = (self.get_sim(self.user_embedding[uu], self.item_embedding[mm]) + u[1] + m[1]) / 3.0
                    if sim > 0.7:
                        mu_list.append([mm, uu, sim])
        mu_list.sort(key = lambda x:x[2], reverse = True)
        mu_list = mu_list[:min(5, len(mu_list))]
        
        if(len(mu_list) == 0):
            return 
        self.outfile.write(str(s_u) + ',' + str(e_m) + '\t' + str(len(mu_list)))
        for mu in mu_list:
            path = ['u' + str(s_u), 'm' + str(mu[0]), 'u' + str(mu[1]), 'm' + str(e_m)]
            self.outfile.write('\t' + '-'.join(path) + ' ' + str(mu[2]))
        self.outfile.write('\n')   
        
    def walk_umtm(self,s_u, e_m):
        limit = 10
        m_list = []
        for m in self.um_dict[s_u]:
            sim = self.get_sim(self.user_embedding[s_u], self.item_embedding[m]) 
            m_list.append([m, sim])
        m_list.sort(key = lambda x:x[1], reverse = True)
        m_list = m_list[:min(limit, len(m_list))]
        
        t_list = []
        for t in self.mt_dict.get(e_m, []):
            t_list.append([t, 1])

        mt_list = []
        for m in m_list:
            for t in t_list:
                mm = m[0]
                tt = t[0]
                if mm in self.mt_dict and tt in self.mt_dict[mm] and  mm != e_m:
                    sim = m[1]
                    if sim > 0.7:
                        mt_list.append([mm, tt, sim])
        mt_list.sort(key = lambda x:x[2], reverse = True)
        mt_list = mt_list[:min(5, len(mt_list))]
        
        if(len(mt_list) == 0):
            return 
        self.outfile.write(str(s_u) + ',' + str(e_m) + '\t' + str(len(mt_list)))
        for mt in mt_list:
            path = ['u' + str(s_u), 'm' + str(mt[0]), 't' + str(mt[1]), 'm' + str(e_m)]
            self.outfile.write('\t' + '-'.join(path))
        self.outfile.write('\n')   

    def walk_uuum(self,s_u, e_m):
        limit = 10
        uf_list = []
        for uf in self.uu_dict[s_u]:
            uf_list.append([uf, 1])
        
        us_list = []
        for us in self.mu_dict.get(e_m, []):
            sim = self.get_sim(self.item_embedding[e_m], self.user_embedding[us])
            us_list.append([us, sim])
        us_list.sort(key = lambda x:x[1], reverse = True)
        us_list = us_list[:limit]
    
        uu_list = []
        for uf in uf_list:
            for us in us_list:
                uff = uf[0]
                uss = us[0]
                if uff in self.uu_dict and uss in self.uu_dict[uff] and  uss != s_u:
                    sim = us[1]
                    if sim > 0.7:
                        uu_list.append([uff, uss, sim])
        uu_list.sort(key = lambda x:x[2], reverse = True)
        uu_list = uu_list[:5]
        
        if(len(uu_list) == 0):
            return 
        self.outfile.write(str(s_u) + ',' + str(e_m) + '\t' + str(len(uu_list)))
        for uu in uu_list:
            path = ['u' + str(s_u), 'u' + str(uu[0]), 'u' + str(uu[1]), 'm' + str(e_m)]
            self.outfile.write('\t' + '-'.join(path) + ' ' + str(uu[2]))
        self.outfile.write('\n')   

    def walk_ummm(self,s_u, e_m):
        limit = 10
        mf_list = []
        for mf in self.um_dict[s_u]:
            sim = self.get_sim(self.item_embedding[mf], self.user_embedding[s_u])
            mf_list.append([mf, sim])
        mf_list.sort(key = lambda x : x[1], reverse = True)
        mf_list = mf_list[:limit]
        
        ms_list = []
        for ms in self.mm_dict.get(e_m, []):
            ms_list.append([ms, 1])
    
        mm_list = []
        for mf in mf_list:
            for ms in ms_list:
                mff = mf[0]
                mss = ms[0]
                if mff in self.mm_dict and mss in self.mm_dict[mff] and  mff != e_m:
                    sim = mf[1]
                    if sim > 0.7:
                        mm_list.append([mff, mss, sim])
        mm_list.sort(key = lambda x:x[2], reverse = True)
        mm_list = mm_list[:5]
        
        if(len(mm_list) == 0):
            return 
        self.outfile.write(str(s_u) + ',' + str(e_m) + '\t' + str(len(mm_list)))
        for mm in mm_list:
            path = ['u' + str(s_u), 'm' + str(mm[0]), 'm' + str(mm[1]), 'm' + str(e_m)]
            self.outfile.write('\t' + '-'.join(path) + ' ' + str(mm[2]))
        self.outfile.write('\n')   

    def walk_mumt(self,start, end):
        path = ['m' + str(start)]
        
        #m - u
        #print start
        if start not in self.mu_dict:
            return None
        index = np.random.randint(len(self.mu_dict[start]))
        u = self.mu_dict[start][index]
        path.append('u' + str(u))
        # u - m
        if u not in self.um_dict:
            return None
        index = np.random.randint(len(self.um_dict[u]))
        m = self.um_dict[u][index]
        path.append('m' + str(m))
        # m - t
        #print path
        if m not in self.mt_dict:
            return None
        if end not in self.mt_dict[m]:
            return None
        path.append('t' + str(end))
        return '-'.join(path)
    
    def walk_mumumt(self,start, end):
        path = ['m' + str(start)]
        
        #m - u
        #print start
        if start not in self.mu_dict:
            return None
        index = np.random.randint(len(self.mu_dict[start]))
        u = self.mu_dict[start][index]
        path.append('u' + str(u))
        # u - m
        if u not in self.um_dict:
            return None
        index = np.random.randint(len(self.um_dict[u]))
        m = self.um_dict[u][index]
        path.append('m' + str(m))

        # m - u
        if m not in self.mu_dict:
            return None
        index = np.random.randint(len(self.mu_dict[m]))
        u = self.mu_dict[m][index]
        path.append('u' + str(u))

        # u - m
        if u not in self.um_dict:
            return None
        index = np.random.randint(len(self.um_dict[u]))
        m = self.um_dict[u][index]
        path.append('m' + str(m))
        
        # m - t
        #print path
        if m not in self.mt_dict:
            return None
        if end not in self.mt_dict[m]:
            return None
        path.append('t' + str(end))
        return '-'.join(path)
    
    def random_walk(self, start):
        path = [self.metapath[0] + start]
        iterator = 0
        k = 1
        while True:
            if k == len(self.metapath):
                iterator += 1
                k = 0
                if iterator == K:
                    return '-'.join(path)

            if k == 0 and self.metapath[k] == 'u':
                pre = path[-1][1:]
                neighbors = self.mu_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif k == 0 and self.metapath[k] == 'm':
                pre = path[-1][1:]
                neighbors = self.um_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
             
            elif self.metapath[k-1] == 'u' and self.metapath[k] == 'm':
                pre = path[-1][1:]
                neighbors = self.um_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
            
            elif self.metapath[k-1] == 'm' and self.metapath[k] == 't':
                pre = path[-1][1:]
                neighbors = self.mt_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif self.metapath[k-1] == 't' and self.metapath[k] == 'm':
                pre = path[-1][1:]
                neighbors = self.tm_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
                
            elif self.metapath[k-1] == 'm' and self.metapath[k] == 'u':
                pre = path[-1][1:]
                neighbors = self.mu_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif self.metapath[k-1] == 'u' and self.metapath[k] == 'a':
                pre = path[-1][1:]
                neighbors = self.ua_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
            elif self.metapath[k-1] == 'a' and self.metapath[k] == 'u':
                pre = path[-1][1:]
                neighbors = self.au_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
            
            elif self.metapath[k-1] == 'u' and self.metapath[k] == 'o':
                pre = path[-1][1:]
                neighbors = self.uo_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
            elif self.metapath[k-1] == 'o' and self.metapath[k] == 'u':
                pre = path[-1][1:]
                neighbors = self.ou_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

    def load_um(self, umfile):
        with open(umfile) as infile:
            for line in infile.readlines():
                u, m = line.strip().split('\t')[:2]
                u, m = int(u), int(m)
                #self.um_list.append([u, m]);
                if u not in self.um_dict:
                    self.um_dict[u] = list()
                self.um_dict[u].append(m)

                if m not in self.mu_dict:
                    self.mu_dict[m] = list()
                self.mu_dict[m].append(u)

    def load_uu(self, uufile):
        with open(uufile) as infile:
            for line in infile.readlines():
                u1, u2 = line.strip().split('\t')[:2]
                u1, u2 = int(u1), int(u2)
                if u1 not in self.uu_dict:
                    self.uu_dict[u1] = list()
                self.uu_dict[u1].append(u2)

                if u2 not in self.uu_dict:
                    self.uu_dict[u2] = list()
                self.uu_dict[u2].append(u1)

    def load_mm(self, mmfile):
        with open(mmfile) as infile:
            for line in infile.readlines():
                m1, m2 = line.strip().split('\t')[:2]
                m1, m2 = int(m1), int(m2)
                if m1 not in self.mm_dict:
                    self.mm_dict[m1] = list()
                self.mm_dict[m1].append(m2)

                if m2 not in self.mm_dict:
                    self.mm_dict[m2] = list()
                self.mm_dict[m2].append(m1)

    def load_uo(self, uofile):
        with open(uofile) as infile:
            for line in infile.readlines():
                u, o = line.strip().split('\t')[:2]
                u, o = int(u), int(o)
                if u not in self.uo_dict:
                    self.uo_dict[u] = list()
                self.uo_dict[u].append(o)

                if o not in self.ou_dict:
                    self.ou_dict[o] = list()
                self.ou_dict[o].append(u)

    def load_mt(self, mtfile):
        with open(mtfile) as infile:
            for line in infile.readlines():
                m, t= line.strip().split('\t')[:2]
                m, t = int(m), int(t)
                if m not in self.mt_dict:
                    self.mt_dict[m] = list()
                self.mt_dict[m].append(t)

                if t not in self.tm_dict:
                    self.tm_dict[t] = list()
                self.tm_dict[t].append(m)

if __name__ == '__main__':
    umfile = '../data/ml-100k.train.rating' 
    uafile = '../data/ml-100k.ua'
    uofile ='../data/ml-100k.uo'
    mtfile = '../data/ml-100k.mt' 
    uufile = '../data/ml-100k.uu_knn_50'
    mmfile = '../data/ml-100k.mm_knn_50'
    walk_num = 5
    K = 1
    metapath = 'umtm' 
    outfile_name = '../data/ml-100k_50.' + metapath + '_' + str(walk_num) + '_' + str(K)
    print('outfile name = ', outfile_name)
    MetapathBasePathSample(uufile = uufile, mmfile = mmfile, umfile = umfile, uafile = uafile, uofile = uofile, mtfile = mtfile,
                           K = K, walk_num = walk_num, metapath = metapath, outfile_name = outfile_name)
