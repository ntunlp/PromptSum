from flask import Flask, render_template, url_for, request, redirect, flash
import numpy as np
import os, os.path
import json
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import random
import pickle
ip_address = "10.97.26.125"
app = Flask(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

text_pegasus = pickle.load(open("data/cnndm_pegasus_50.pkl", "rb"))
print(text_pegasus.keys())
print(len(text_pegasus["src"]))
text_promptsum = pickle.load(open("data/cnndm_promptsum_50.pkl", "rb"))
print(text_promptsum.keys())
print(len(text_promptsum["src"]))
text = {}
text['src'] = text_pegasus['src']
text['pegasus'] = text_pegasus['model']
text['promptsum'] = text_promptsum['model']

# Should load all existing users
print(os.listdir('users/'))
number_user = len(os.listdir('users/'))
number_per_user = 50 # number of data points to evaluate
user_index = 0


class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])
    email = TextField('Email:', validators=[validators.required(), validators.Length(min=6, max=35)])
    password = TextField('Password:', validators=[validators.required(), validators.Length(min=3, max=35)])

    @app.route("/", methods=['GET', 'POST'])
    def hello():
        if len(os.listdir('users')) == 0:
            global user_index
        else:
            print("Not empty")
            
        form = ReusableForm(request.form)
        # print form.errors
        name_list = [i.split('_')[0] for i in os.listdir('users/')]
        # if len(name_list)==0:
        #     global index
        if request.method == 'POST':
            name = request.form['name']
            if name not in name_list:

                uid = "{}_{}".format(name, user_index)
                os.mkdir('users/%s' % uid)
                os.mkdir('users/%s/result' % uid)
                user_index = user_index + 1
            else:
                uid = os.listdir('users')[name_list.index(name)]
                user_index = int(os.listdir('users')[name_list.index(name)].split('_')[1])

            return redirect('http://' + ip_address + ':5000/%s' % uid)
        if form.validate():
            # Save the comment here.
            flash('Thanks for registration ' + name)
        else:
            flash('Error: All the form fields are required. ')

        return render_template('index.html', form=form)


@app.route('/thanks')
def thanks():
    return render_template('thanks.html')


@app.route('/<string:uid>', methods=['GET', 'POST'])
def start(uid):
    if request.method == 'POST':

        preference = request.form['preference']
        reason = request.form['reason']
        # print lan_ranks
        reversed_ = request.form['reversed_']
        # print reversed_
        postid =request.form['postid']

        result = {
            'preference': preference,
            'reason': reason,
            'reversed_': reversed_
        }

        # Save results
        with open('users/%s/result/%d.json' % (uid, int(postid)), 'w') as outfile:
            json.dump(result, outfile)

    if uid in os.listdir('users/'):
        user_result_dir = 'users/%s/result/' % uid
        user_index = int(uid.split("_")[1])
        postid = len(os.listdir(user_result_dir))  # Start from next question if the user exists
    else:
        print('uid: ',uid)
        user_index = int(uid.split("_")[1])
        postid = 0

    if postid >= number_per_user: # Stop when all images have been evaluated
        return redirect('http://'+ip_address+':5000/thanks')
    cap = []
    source = text["src"][postid]
    pegasus = text["pegasus"][postid].replace("<n>", " ")
    promptsum = text["promptsum"][postid].replace("<n>", " ")

    cap.append(pegasus)
    cap.append(promptsum)

    orders =[[0, 1], [1, 0]]
    new_order = random.choice(orders)
    if new_order == [0, 1]:
        reversed_ = "False"
    else:
        reversed_ = "True"

    cap1 = cap[new_order[0]]
    cap2 = cap[new_order[1]]

    prog = postid

    return render_template('start_ranking.html', source=source, cap1=cap1, cap2=cap2,
                           postid=postid, reversed_=reversed_, question_type=0, prog=prog)  # Continue to next question
