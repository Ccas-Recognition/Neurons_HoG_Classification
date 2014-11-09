cd ..
If Not Exist build mkdir build
cd build
If Not Exist bin mkdir bin
cd ..\vs2013
copy Release\Neurons_HoG_Classification.exe ..\build\bin\task2.exe

