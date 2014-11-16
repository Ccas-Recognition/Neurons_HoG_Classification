cd ..
If Exist ..\..\repo\neuronsrecognition\hog\ ( 
::copy src\HOG.h ..\..\repo\neuronsrecognition\hog\HOG.h
::copy src\classifier.h ..\..\repo\neuronsrecognition\hog\classifier.h
::copy vs2013\liblinear\linear.h ..\..\repo\neuronsrecognition\hog\linear.h
::copy vs2013\liblinear\tron.h ..\..\repo\neuronsrecognition\hog\tron.h
::copy src\HOG_Functor.h ..\..\repo\neuronsrecognition\hog\HOG_Functor.h
::copy src\consts.h ..\..\repo\neuronsrecognition\hog\consts.h
::copy vs2013\Debug\Neurons_HoG_Classification_Lib.lib ..\..\repo\neuronsrecognition\hog\hogd.lib

::copy model_binary.txt ..\..\repo\neuronsrecognition\hog\hog_model.txt
)