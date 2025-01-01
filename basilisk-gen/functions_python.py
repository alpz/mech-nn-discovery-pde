import numpy as np
import os

def ReadDataFile(filename):
	fid = open(filename, 'rb')

	# Reading mesh dimensions from mesh
	
	read_char = chr(np.fromfile(fid, np.ubyte, count=1)[0])
	read_string = read_char
	current_field_name = "%s" % (read_char)
	while( True ):
		read_char = chr(np.fromfile(fid, np.ubyte, count=1)[0])
		if( read_char=="\n" ):
			break
		read_string += read_char


	nx = int(read_string.split(" ")[0])
	ny = int(read_string.split(" ")[1])
	num_cells = nx*ny


	# Reading the names of fields from the file header
	field_names = []
	read_char = chr(np.fromfile(fid, np.ubyte, count=1)[0])
	current_field_name = "%s" % (read_char)
	while( True ):
		read_char = chr(np.fromfile(fid, np.ubyte, count=1)[0])

		if( read_char==";" ):
			field_names.append(current_field_name)
			current_field_name = ""
		elif( read_char!="\n" ):
			current_field_name += "%s" % (read_char)
		else: # read_char==\n
			break
	field_names.append(current_field_name)


	# Reading the binary file
	data_array = np.fromfile(fid, np.double)
	fid.close()


	# Breaking the big chunk of data into separate matrices for each field
	fields = {}
	for i, field_name in enumerate(field_names):
		temp_array = data_array[i*num_cells:(i+1)*num_cells]
		fields[field_name] = temp_array.reshape((nx, ny))
	
	return fields


def ReadAllDataFilesInFolder(foldername):

	total_list_files = os.listdir(foldername)
	list_files = []
	for file in total_list_files:
		#if( file.startswith("Interface") ):
		if( file.startswith("Mesh") ):
			#list_files.append( int(file[11:-4]) )
			list_files.append( int(file[10:-4]) )
	list_files = np.sort(list_files)

	number_snapshots = len(list_files)
	print(number_snapshots)

	# number_snapshots = 5

	for i_snapshot, simulation_step in enumerate(list_files):

		filename = "%s/MeshDump-N%d.bin" % (foldername, simulation_step)
		print(filename)
		snapshot_data = ReadDataFile(filename)

		if( i_snapshot==number_snapshots ):
			break

		print("Reading data from files. Timestep %d out of %d" % (i_snapshot, number_snapshots-1))

		field_names = list(snapshot_data.keys())
		
		# In the first iteration, we initialize the data dictionary
		if( i_snapshot==0 ):
			nx = snapshot_data[field_names[0]].shape[0]
			ny = snapshot_data[field_names[0]].shape[1]

			fields_time = {}
			for f in field_names:
				fields_time[f] = np.zeros(shape=(nx, ny, number_snapshots))

		for f in field_names:
			fields_time[f][:, :, i_snapshot] = snapshot_data[f]

	print("\n\n")
	return fields_time, list_files