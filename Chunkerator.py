import nltk, types
from nltk import tag, chunk

rules = """
	NP: {<DT>?<JJ>*<NN.*>}
	NP: {<PRP>}
"""

class Chunkerator:
	
	def __init__(self, rules):
		self.chunkParser = nltk.RegexpParser(rules)
		self.chunksSeen = []
		
	def initial_tag(self,sentence):
		'''
		Uses the basic nltk tagger to assign tags.
		'''
		sent = sentence.split()
		return tag.pos_tag(sent)
	
	def initial_chunk(self,tagged_sent):
		tree =  self.chunkParser.parse(tagged_sent)
		return tree
	
	def merge_chunk(self,tup):
		'''
		Reads in tuple from tree, flattens it out as atomic word.
		'''
		out = ''
		for z in tup:
			out+=(str(z[0])+str('_'))
		out = out[0:-1]
		self.chunksSeen.append(out)
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
		return output
		
	def chunk_sent(self,sentence):
		tagged_sent = self.initial_tag(sentence)
		tree = self.initial_chunk(tagged_sent)
		out_sent = self.remake_chunked_sent(tree)
		return out_sent

