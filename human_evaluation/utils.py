import urllib2
import HTMLParser
import codecs
import json


class HTMLtoJSONParser(HTMLParser.HTMLParser):
    def __init__(self, raise_exception=True):
        HTMLParser.HTMLParser.__init__(self)
        self.doc = {}
        self.path = []
        self.cur = self.doc
        self.line = 0
        self.raise_exception = raise_exception

    @property
    def json(self):
        return self.doc

    @staticmethod
    def to_json(content, raise_exception=True):
        parser = HTMLtoJSONParser(raise_exception=raise_exception)
        parser.feed(content)
        return parser.json

    def handle_starttag(self, tag, attrs):
        self.path.append(tag)
        attrs = {k: v for k, v in attrs}
        if tag in self.cur:
            if isinstance(self.cur[tag], list):
                self.cur[tag].append({"__parent__": self.cur})
                self.cur = self.cur[tag][-1]
            else:
                self.cur[tag] = [self.cur[tag]]
                self.cur[tag].append({"__parent__": self.cur})
                self.cur = self.cur[tag][-1]
        else:
            self.cur[tag] = {"__parent__": self.cur}
            self.cur = self.cur[tag]

        for a, v in attrs.items():
            self.cur["#" + a] = v
        self.cur[""] = ""

    def handle_endtag(self, tag):
        if tag != self.path[-1] and self.raise_exception:
            raise Exception("html is malformed around line: {0} (it might be because of a tag <br>, <hr>, <img .. > not closed)".format(self.line))
        del self.path[-1]
        memo = self.cur
        self.cur = self.cur["__parent__"]
        self.clean(memo)

    def handle_data(self, data):
        self.line += data.count("\n")
        if "" in self.cur:
            self.cur[""] += data

    def clean(self, values):
        keys = list(values.keys())
        for k in keys:
            v = values[k]
            if isinstance(v, str):
                # print ("clean", k,[v])
                c = v.strip(" \n\r\t")
                if c != v:
                    if len(c) > 0:
                        values[k] = c
                    else:
                        del values[k]
        del values["__parent__"]


def html2txt():
    from bs4 import BeautifulSoup

    f = codecs.open('/home/jxgu/github/unparied_im2text_jxgu/eval_results/caption_zh_en.html', 'r')
    # page = urllib2.urlopen('/home/jxgu/github/unparied_im2text_jxgu/eval_results/caption_zh_en.html')
    content = '<html><body><div class="an_example"><p>one paragraph</p></div></body></html>'

    soup = BeautifulSoup(f.read(), 'html.parser')
    js = HTMLtoJSONParser.to_json(soup.prettify())
    # print(soup.prettify())
    #with open('caption_zh_en.json', 'w') as outfile:
    #    json.dump(js, outfile)


def text2json():
    import random
    out = open('/home/jxgu/github/unparied_im2text_jxgu/eval_results/caption_zh_en.txt', 'r')
    lines = out.readlines()
    out = []
    explanation = {}
    for i in range(len(lines) / 5):
        '''
        tmp = {}
        tmp['file_path'] = lines[i * 5].replace('\n','')
        tmp['id'] = lines[i * 5 + 1].replace('\n','')
        tmp['image_id'] = lines[i * 5 + 2].replace('\n','')
        tmp['zh'] = lines[i * 5 + 3].replace('\n','')
        tmp['en'] = lines[i * 5 + 4].replace('\n','')
        '''
        tmp = [int(lines[i * 5].replace('\n','').split('/')[-1].split('_')[-1].replace('.jpg', '')), 0]
        explanation[str(int(lines[i * 5].replace('\n','').split('/')[-1].split('_')[-1].replace('.jpg', '')))] = lines[i * 5 + 4].replace('\n','')
        out.append(tmp)
    list_of_random_items = random.sample(range(1572), 327)
    for i in list_of_random_items:
        out.append(out[i])
    directory = '/home/jxgu/github/online_eval_tools/main/users'
    for i in range(len(out)/100):
        with open(directory + '/u' + str(i) + '/im_c_list.json', 'w') as outfile:
            json.dump(out[i*100:i*100+100], outfile)
    with open('/home/jxgu/github/online_eval_tools/main/coco_caption_human_eval.json', 'w') as outfile:
        json.dump(explanation, outfile)

def gt2json():
    import random
    imgs = json.load(open('/home/jxgu/github/im2text_jxgu/pytorch/data/mscoco/dataset_coco.json', 'r'))
    imgs = imgs['images']
    out = {}
    for i, img in enumerate(imgs):
        out[str(img['cocoid'])] = img['sentences'][random.randint(0,4) ]['raw']

    json.dump(out, open('/home/jxgu/github/online_eval_tools/main/coco_caption_human_eval_gt.json', 'w'))

gt2json()
