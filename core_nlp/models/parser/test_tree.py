
from core_nlp.models.parser.features import FeatureMapper

from core_nlp.data.phrase_tree import PhraseTree
fm = FeatureMapper.load_json('/Users/qiwang/python-space/nju_nlp_tools/testdata/toy.vocab.json')
test_trees = PhraseTree.load_trees('/Users/qiwang/python-space/nju_nlp_tools/testdata/toy.clean')
#test_trees[0].rotate_tree()
test_trees[0].draw_tree('tree.png')