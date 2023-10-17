from django.shortcuts import render
from django.shortcuts import render
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pyvis.network import Network

model_name = "Babelscape/rebel-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.half()

def get_text(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')
        # We need to use the tokenizer manually since we need special tokens.
        extracted_text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(input_text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])
        # Function to parse the generated text and extract the triplets
        def extract_triplets(text):
            triplets = []
            relation, subject, relation, object_ = '', '', '', ''
            text = text.strip()
            current = 'x'
            for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
                if token == "<triplet>":
                    current = 't'
                    if relation != '':
                        triplets.append({'head': subject.strip(), 'edge': relation.strip(),'tail': object_.strip()})
                        relation = ''
                    subject = ''
                elif token == "<subj>":
                    current = 's'
                    if relation != '':
                        triplets.append({'head': subject.strip(), 'edge': relation.strip(),'tail': object_.strip()})
                    object_ = ''
                elif token == "<obj>":
                    current = 'o'
                    relation = ''
                else:
                    if current == 't':
                        subject += ' ' + token
                    elif current == 's':
                        object_ += ' ' + token
                    elif current == 'o':
                        relation += ' ' + token
            if subject != '' and relation != '' and object_ != '':
                triplets.append({'head': subject.strip(), 'edge': relation.strip(),'tail': object_.strip()})
            # print('&&&&&&&&&&&&&&&&&&&&&',triplets)
            return triplets
        extracted_triplets = extract_triplets(extracted_text[0])
        print('###############',extracted_triplets)

        # create a new network        
        graph = Network(directed=True, height="550px", width="100%", bgcolor="#FFFFFF", font_color="black")
        # add nodes         
        for i in range(len(extracted_triplets)):
            graph.add_node(extracted_triplets[i]['head'],shape="circle", label=extracted_triplets[i]['head'])
            graph.add_node(extracted_triplets[i]['tail'],shape="circle", label=extracted_triplets[i]['tail'])
            # add edges             
            graph.add_edge(extracted_triplets[i]['head'],extracted_triplets[i]['tail'],label=extracted_triplets[i]['edge'])
        graph.repulsion(
                node_distance=400,
                central_gravity=0.2,
                spring_length=200,
                spring_strength=0.05,
                damping=0.09)
        graph_html = graph.generate_html()
        return render(request,'index.html',{'graph_html': graph_html,"input_text":input_text})
    else:
        return render(request,'index.html')


