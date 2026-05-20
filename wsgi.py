from _ml_backend import app

if __name__ == '__main__' :
   print('wsgi catched app of ml backend py and go run!!!')
   app.run(host='0.0.0.0', port=5001, debug=False)
