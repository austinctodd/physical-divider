Once you checked out the project, you need to configure it. This is done
as follows:
There is a file
  
    genericConf.py

in the root directory, that contains all the configuration. Copy this file
to conf.py and open conf.py for editing:

    $ cp genericConf.py conf.py
    $ vim conf.py

At the beginning of this file you find the section for site specific
configuration. Set all the variables in there to values appropriate for your   
site. Then save and exit the editor.

To run the tests, call tests/runAllTests.py.
