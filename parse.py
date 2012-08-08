import doxament
import nltk
import types
from nltk import tag, chunk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from itertools import combinations
rules = """
    NP: {<DT>?<JJ>*<NN.*>}
    NP: {<PRP>}
"""
 
class Doxament:
    relations = []

    def __init__(self, relations):
        self.relations = relations

    def query(self, qdox):
        '''
        Query the first doxament with the second
        doxament for consistency/coverage
        '''
        total = len(qdox.relations)
        found = 0
        contras = []
        
        for r in qdox.relations:
            found += 1 if r in self.relations else 0
            contra_rel = r.flip()
            if contra_rel in self.relations:
                contras.append(contra_rel) 
                found -= 1

        score = float(found) / total
        return score, contras

           
class Document:
    '''
    A class that represents unprocessed text.
    May later include metadata.
    '''

    text = ''

    def __init__(self,text):
        self.text = text
        
    def __str__(self):
        return self.text

    def to_dox(self):
        return Doxament(Parser(self).parse_relations())

    def tokenize(self):
        '''
        Returns a list of tokenized sentences
        '''
        sentence_tokenizer = PunktSentenceTokenizer()
        sentences = sentence_tokenizer.sentences_from_text(self.text)
        sentences = [sentence.split() for sentence in sentences]
        sentences = [[word.strip(",.?!") for word in sentence]
                     for sentence in sentences]
        return sentences

class Parser:
    '''
    Class responsible for parsing a Document into
    a collection of Relations.
    '''
    doc = ''

    # initialize with a Document
    def __init__(self, doc):
        self.doc = doc

    def parse_relations(self):
        sentences = self.doc.tokenize()
        sentences,chunker = self.preprocess(sentences)

        relations = []
        for sentence in sentences:            
            relations.extend(self.parse_sentence(sentence,chunker))

        return relations

    def preprocess(self, sentences):
        '''
        Takes a list of strings representing sentences.
        Returns list of processed tokens, suitable for
        converting to Relations.
        '''
        post = []
        c = Chunkerator(rules)
        for sentence in sentences:
            # part of speech
            sent = c.chunk_sent(sentence)
            # pronoun resolution
            ps = self.neg_scope(sent)
            ps = [w for w in ps if w.lower() not in stopwords.words("english")]
            post.append(ps)

        return post,c

    def neg_scope(self, sentence):
        neg_words = ['not','never', 'isn\'t','was\'nt','hasn\'t']
        for ii in xrange(len(sentence)):
            if sentence[ii] in neg_words:
            #should really go to next punctuation pt, complementizer(?), clause-boundary 
                for jj in range(ii+1,len(sentence)):
                    sentence[jj] = 'neg_%s' % sentence[jj]
        
        return sentence

    def parse_sentence(self,sentence,chunkerator):
            pairs = combinations(sentence,2)
            relations = [self.make_relation(p) for p in pairs]
            for chunk in chunkerator.chunksSeen:
                g = Relation(True,chunk,chunkerator.chunksSeen[chunk])
                relations.append(g)
                splitchunkpairs = combinations(chunk.split('_'),2)
                chunkRels = [self.make_relation(x) for x in splitchunkpairs]
                relations.extend(chunkRels)
            print relations
            return relations

    def make_relation(self,pair):
        co = True
        item1,item2 = pair

        if self.is_neg(item1):
            item1 = self.strip_neg(item1)
            co = not co

        if self.is_neg(item2):
            item2 = self.strip_neg(item2)
            co = not co

        return Relation(co,item1,item2)

    def strip_neg(self,word):
        if word[0:4] == "neg_":
            return word[4:]
        else:
            return word

    def is_neg(self,word):
        return word[0:4] == "neg_"


class Relation:
    co = True
    item1 = ''
    item2 = ''

    def __init__(self,co,item1,item2):
        self.co = co
        self.item1 = item1
        self.item2 = item2

    def flip(self):
        return Relation(not self.co, self.item1, self.item2)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.co == other.co:
                return ((syno(self.item1,other.item1) and
                         syno(self.item2,other.item2)) or
                        (anto(self.item1,other.item1) and
                         anto(self.item2,other.item2)))
            else:
                return ((syno(self.item1,other.item1) and
                         anto(self.item2,other.item2)) or
                        (anto(self.item1,other.item1) and
                         syno(self.item2,other.item2)))
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str((self.co, self.item1, self.item2))

    def __repr__(self):
        return str((self.co, self.item1, self.item2))


class Chunkerator:

    def __init__(self, rules):
        self.chunkParser = nltk.RegexpParser(rules)
        self.chunksSeen = dict()
        self.ent_count = 0

    def initial_tag(self,sentence):
        '''
        Uses the basic nltk tagger to assign tags.
        '''
        #sent = sentence.split()
        return tag.pos_tag(sentence)

    def initial_chunk(self,tagged_sent):
        tree =  self.chunkParser.parse(tagged_sent)
        return tree

    def merge_chunk(self,tup):
        '''
        Reads in tuple from tree, flattens it out as atomic word.
        Adds chunk to self.chunksSeen with entityID
        '''
        out = ''
        for z in tup:
            out+=(str(z[0])+str('_'))
        out = out.rstrip('_')
        self.chunksSeen[out] ='ent'+str(self.ent_count)
        self.ent_count += 1 
        return out

    def remake_chunked_sent(self,tree):
        '''
        Reconstitutes the original sentence with chunks treated as atomic.
        '''
        output = '' 
        for xx in tree:
            if type(xx[0]) == types.TupleType:      
                output+= str(self.merge_chunk(xx))+' '
            else:
                output+= str(xx[0])+' '
        output.rstrip()
        return output.split()

    def assign_entIDs(self, sentence):
        for x in xrange(0,len(sentence)):
            if sentence[x] in self.chunksSeen:
                sentence[x] = self.chunksSeen[sentence[x]]
        return sentence

    def chunk_sent(self,sentence):
        tagged_sent = self.initial_tag(sentence)
        tree = self.initial_chunk(tagged_sent)
        out_sent = self.remake_chunked_sent(tree)
        out_sent = self.assign_entIDs(out_sent)
        return out_sent
                
def syno(item1,item2):
    #should test synsets, this is dummy
    return item1 == item2

def anto(item1,item2):
    #should test antonyms, this is dummy
    return False

def aggregate_lemmas(word,relation):
    '''
    Generates a list of synonyms/antonyms for :word: 
    '''
    lems = set()
    if relation == "synonym":
        sets = [syn.lemmas for syn in wn.synsets(word)]
    elif relation == "antonym":
        sets = [syn.lemmas for syn in wn.synsets(word)]
        sets = list(itertools.chain(*sets))
        sets = [x.antonyms() for x in sets]
        sets = [x for x in sets if x]
        
    sets = list(itertools.chain(*sets))
    sets = [lem.name for lem in sets]
    for x in sets:
        lems.add(x)
    return lems

senttt = 'The man has eaten some green cheese.'.split()
text1 = "Today was a good day.  Yesterday was a bad day."
doc1 = Document(text1)
print doc1
dox1 = doc1.to_dox()