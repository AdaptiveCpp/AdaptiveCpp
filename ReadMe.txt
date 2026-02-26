**********************************************
* 	IncrediBuild Software - Make Sample	     *
**********************************************

This sample project demonstrates a simple application build using make and
the Automatic Interception Interface.

Notes:
1. You must have an operable make environment set up in order to run this
   sample.

2. The location of the make executable should be defined in the path to allow
   initiation from any folder.

3. Note that the example is currently configured to use the MS Visual Studio
   2010 compiler. If you have a different Visual Studio version installed,
   open RunMakeSample.bat and replace the following line:
   call "%VS100COMNTOOLS%vsvars32.bat"
   with either:
   call "%VS90COMNTOOLS%vsvars32.bat"       (for MS Visual Studio 2008)
   Or: 
   vsvars32.bat of any otther Visual studio version.
