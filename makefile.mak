OBJS = demo_main.obj demo_c0.obj demo_c1.obj demo_c2.obj demo_c3.obj demo_c4.obj demo_c5.obj demo_c6.obj demo_c7.obj demo_c8.obj demo_c9.obj

CL=cl.exe
LINK=link.exe
APP=demo.exe

ALL : $(APP)

.SUFFIXES = .c .obj
%.obj : %.c
	$(CL) /nologo /Fo$@ /c $<

$(APP) : $(OBJS)
	LINK /out:$(APP) /nologo $(OBJS)
