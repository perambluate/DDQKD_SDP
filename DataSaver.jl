module DataSaver
using DelimitedFiles

export saveData

function saveData(data, file::String; header::String="", mode::String="w")
    if isfile(file) && occursin("w", mode)
        println("The file '", file, "' already exists, do u want to overwrite?[Y/y]")
        input = readline()
        if !(input == "y" || input == "Y")
            return
        end
    end

    open(file, mode) do io
        if isfile(file)
            write(io, "\n")
        end
        if !isempty(header)
            write(io, "#", header, "\n")
        end
        writedlm(io, data, ',')     
    end
end

end #module