####    Lazy data descriptors    ####

abstract type AbstractLazyData end

"""
	LazyArray(filnam, mtime, begpos, format, fields, keywds)

	Lazy image-like FITS data.  Values are read from disk on demand.
"""
struct LazyArray{T, N} <: DiskArrays.AbstractDiskArray{T, N}
    "File name"
    filnam::String
    "Modification time"
    mtime::Float64
    "Beginning file position"
    begpos::Int64
    "HDU type"
    hdu_type::Type
    "Array format"
    format::DataFormat
    "Field formats"
    fields::Any
    "HDU keywords"
    keywds::NamedTuple
end

struct LazyStructuredData <: AbstractLazyData
    filnam::String
    mtime::Float64
    begpos::Int64
    hdu_type::Type
    format::DataFormat
    fields::Vector
    keywds::NamedTuple
    record::Bool
end

struct LazyFieldArray{T, N} <: DiskArrays.AbstractDiskArray{T, N}
    parent::LazyStructuredData
    fields::Vector
end

const LAZY_STRUCTURED_INTERNALS =
    (:filnam, :mtime, :begpos, :hdu_type, :format, :fields, :keywds, :record)

function LazyArray(
        filnam::AbstractString, mtime::Real, begpos::Integer,
        hdu_type::Type, format::DataFormat, fields, keywds::NamedTuple
    )

    T = lazy_eltype(format.type, fields, get(keywds, :scale, true))
    N = length(format.shape)
    return LazyArray{T, N}(
        String(filnam), Float64(mtime), Int64(begpos), hdu_type,
        format, fields, keywds
    )
end

function LazyArray(
        filnam::AbstractString, mtime::Real, begpos::Integer,
        format::DataFormat, fields, keywds::NamedTuple
    )

    return LazyArray(filnam, mtime, begpos, Primary, format, fields, keywds)
end

function lazydata(
        hdu_type::Type, filnam::AbstractString, mtime::Real,
        begpos::Integer, format::DataFormat, fields, keywds::NamedTuple
    )

    return if hdu_type in (Primary, Image, Conform)
        LazyArray(filnam, mtime, begpos, hdu_type, format, fields, keywds)
    elseif hdu_type in (Random, Table, Bintable)
        LazyStructuredData(
            String(filnam), Float64(mtime), Int64(begpos),
            hdu_type, format, Vector(fields), keywds, get(keywds, :record, false)
        )
    else
        LazyStructuredData(
            String(filnam), Float64(mtime), Int64(begpos),
            hdu_type, format, Vector(fields), keywds, get(keywds, :record, false)
        )
    end
end

####    Shared lazy source helpers    ####

lazy_source(data::Union{LazyArray, LazyStructuredData}) =
    (data.filnam, data.mtime, data.begpos, data.format)

lazy_format(data::Union{LazyArray, LazyStructuredData}) = data.format
lazy_fields(data::Union{LazyArray, LazyStructuredData}) = data.fields

function open_lazy_source(data::Union{LazyArray, LazyStructuredData})
    isfile(data.filnam) || error("Lazy FITS source no longer exists: $(data.filnam)")
    mtime = stat(data.filnam).mtime
    mtime == data.mtime ||
        error("Lazy FITS source has changed since it was read: $(data.filnam)")
    return open(data.filnam, "r")
end

function read_bigendian(io::IO, type::Type)
    return type <: Bool ? read(io, type) : ntoh(read(io, type))
end

function read_bigendian(io::IO, type::Type, n::Integer)
    return ntoh.(reinterpret(type, read(io, sizeof(type) * n)))
end

function read_bigendian_at(filnam::AbstractString, pos::Integer, type::Type)
    return open(filnam, "r") do io
        seek(io, pos)
        read_bigendian(io, type)
    end
end

# Whether the field's PSCAL/PZERO transform is the identity (PSCAL=1,
# PZERO=0). Identity scaling must NOT promote integer columns to float
# — the FITS standard defaults to storing PSCAL=1, PZERO=0 as floats
# even for integer columns, and naively applying `zero + scale*value`
# converts every integer column to Float32 / Float64.
function _is_identity_scaling(field)
    return hasproperty(field, :zero) && hasproperty(field, :scale) &&
        !isnothing(getproperty(field, :zero)) && !ismissing(getproperty(field, :zero)) &&
        !isnothing(getproperty(field, :scale)) && !ismissing(getproperty(field, :scale)) &&
        iszero(getproperty(field, :zero)) && isone(getproperty(field, :scale))
end

function _has_active_scaling(field, scale::Bool)
    return scale && hasproperty(field, :zero) && hasproperty(field, :scale) &&
        !isnothing(getproperty(field, :zero)) && !ismissing(getproperty(field, :zero)) &&
        !_is_identity_scaling(field)
end

function scale_value(value, field, scale::Bool)
    return if _has_active_scaling(field, scale)
        getproperty(field, :zero) + getproperty(field, :scale) * value
    else
        value
    end
end

function lazy_eltype(type::Type, field, scale::Bool)
    return if type <: AbstractString
        String
    elseif type <: BitVector
        BitVector
    elseif _has_active_scaling(field, scale)
        typeof(getproperty(field, :zero) + getproperty(field, :scale) * zero(type))
    else
        type
    end
end

lazy_eltype(type::Type, fields::AbstractVector, scale::Bool) =
    lazy_eltype(type, first(fields), scale)

function scaled_values(values, field, scale::Bool)
    return if _has_active_scaling(field, scale)
        getproperty(field, :zero) .+ getproperty(field, :scale) .* values
    else
        values
    end
end

is_unit_ranges(ranges) = all(r -> r isa AbstractUnitRange, ranges)

function contiguous_linear_span(shape::Tuple, ranges)
    is_unit_ranges(ranges) || return nothing
    lins = LinearIndices(shape)
    firsts = map(first, ranges)
    lasts = map(last, ranges)
    first_index = lins[firsts...]
    last_index = lins[lasts...]
    n = prod(length, ranges)
    return last_index - first_index + 1 == n ? (first_index:last_index) : nothing
end

data_nbytes(format::DataFormat) = sizeof(format.type) * format.leng
data_padded_nbytes(format::DataFormat) = BLOCKLEN * div(data_nbytes(format), BLOCKLEN, RoundUp)

function copy_lazy_data_block(io::IO, data::Union{LazyArray, LazyStructuredData})
    src = open_lazy_source(data)
    return try
        seek(src, data.begpos)
        remaining = data_padded_nbytes(data.format)
        buffer = Vector{UInt8}(undef, min(remaining, 1024 * 1024))
        while remaining > 0
            n = min(length(buffer), remaining)
            readbytes!(src, buffer, n)
            write(io, n == length(buffer) ? buffer : @view(buffer[1:n]))
            remaining -= n
        end
    finally
        close(src)
    end
end

function materialize_lazy_data(data::LazyArray)
    io = open_lazy_source(data)
    return try
        seek(io, data.begpos)
        read(io, data.hdu_type, data.format, data.fields; data.keywds...)
    finally
        close(io)
    end
end

function materialize_lazy_data(data::LazyStructuredData)
    return if data.record
        [read_lazy_row(data, j) for j in 1:structured_nrows(data)]
    else
        (; [name => read(data[name]) for name in keys(data)]...)
    end
end

Base.read(data::Union{LazyArray, LazyStructuredData}) = materialize_lazy_data(data)
Base.collect(data::LazyArray) = read(data)

####    Image-like lazy arrays    ####

Base.size(data::LazyArray) = data.format.shape
Base.axes(data::LazyArray) = map(Base.OneTo, size(data))
DiskArrays.haschunks(::LazyArray) = DiskArrays.Unchunked()

function image_linear_value(data::LazyArray, index::Integer)
    io = open_lazy_source(data)
    return try
        image_linear_value(io, data, index)
    finally
        close(io)
    end
end

function image_linear_value(io::IO, data::LazyArray, index::Integer)
    seek(io, data.begpos + (index - 1) * sizeof(data.format.type))
    value = read_bigendian(io, data.format.type)
    return scale_value(value, data.fields, get(data.keywds, :scale, true))
end

function DiskArrays.readblock!(data::LazyArray, out, ranges::OrdinalRange...)
    ndims(data) == length(ranges) || error("Number of indices is not correct")
    lins = LinearIndices(size(data))
    io = open_lazy_source(data)
    try
        span = contiguous_linear_span(size(data), ranges)
        if !isnothing(span)
            seek(io, data.begpos + (first(span) - 1) * sizeof(data.format.type))
            values = read_bigendian(io, data.format.type, length(span))
            out .= reshape(
                scaled_values(
                    values, data.fields,
                    get(data.keywds, :scale, true)
                ), size(out)
            )
        elseif read_image_firstdim_spans!(io, data, out, ranges, lins)
        else
            for ci in CartesianIndices(out)
                index = ntuple(j -> ranges[j][ci[j]], ndims(data))
                out[ci] = image_linear_value(io, data, lins[index...])
            end
        end
    finally
        close(io)
    end
    return out
end

function read_image_firstdim_spans!(io::IO, data::LazyArray, out, ranges, lins)
    ndims(data) > 1 || return false
    first(ranges) isa AbstractUnitRange || return false
    is_unit_ranges(Base.tail(ranges)) || return false

    span_length = length(first(ranges))
    rest_ranges = Base.tail(ranges)
    rest_shape = Base.tail(size(out))
    scale = get(data.keywds, :scale, true)

    for restci in CartesianIndices(rest_shape)
        rest = ntuple(j -> rest_ranges[j][restci[j]], length(rest_shape))
        first_index = lins[first(first(ranges)), rest...]
        seek(io, data.begpos + (first_index - 1) * sizeof(data.format.type))
        values = read_bigendian(io, data.format.type, span_length)
        view(out, :, Tuple(restci)...) .= scaled_values(values, data.fields, scale)
    end
    return true
end

####    Structured lazy data    ####

function field_name(field)
    return Symbol(rstrip(String(field.name)))
end

function field_groups(data::LazyStructuredData)
    names = field_name.(data.fields)
    return [name => findall(==(name), names) for name in unique(names)]
end

Base.keys(data::LazyStructuredData) = first.(field_groups(data))

function Base.propertynames(data::LazyStructuredData, private::Bool = false)
    names = Tuple(keys(data))
    return private ? (LAZY_STRUCTURED_INTERNALS..., names...) : names
end

Base.hasproperty(data::LazyStructuredData, name::Symbol) =
    name in propertynames(data, true)

function Base.length(data::LazyStructuredData)
    return data.record ? structured_nrows(data) : length(keys(data))
end

function structured_nrows(data::LazyStructuredData)
    return if data.hdu_type == Random
        data.format.group
    else
        length(data.format.shape) >= 2 ? data.format.shape[2] : 0
    end
end

function field_shape(data::LazyStructuredData, fields::Vector)
    return if data.hdu_type == Random
        if length(fields) > 1
            (data.format.group, length(fields))
        else
            field = first(fields)
            field.leng == 1 ? (data.format.group,) : (data.format.group, field.shape...)
        end
    elseif data.hdu_type == Bintable
        field = first(fields)
        if !isnothing(field.pntr) || field.type <: AbstractString ||
                field.type <: BitVector || field.leng == 1
            (data.format.shape[2],)
        else
            (data.format.shape[2], field.leng)
        end
    else
        (data.format.shape[2],)
    end
end

function field_array_eltype(data::LazyStructuredData, fields::Vector)
    field = first(fields)
    return if data.hdu_type == Table
        lazy_eltype(field.type, field, get(data.keywds, :scale, true))
    elseif data.hdu_type == Bintable && !isnothing(field.pntr)
        Vector{field.type}
    else
        lazy_eltype(field.type, field, get(data.keywds, :scale, true))
    end
end

function LazyFieldArray(data::LazyStructuredData, fields::Vector)
    T = field_array_eltype(data, fields)
    N = length(field_shape(data, fields))
    return LazyFieldArray{T, N}(data, fields)
end

function Base.getindex(data::LazyStructuredData, name::Union{Symbol, AbstractString})
    sym = name isa Symbol ? name : Symbol(rstrip(String(name)))
    groups = Dict(field_groups(data))
    haskey(groups, sym) || throw(KeyError(name))
    return LazyFieldArray(data, data.fields[groups[sym]])
end

function Base.getindex(data::LazyStructuredData, index::Integer)
    return if data.record
        read_lazy_row(data, index)
    else
        data[collect(keys(data))[index]]
    end
end

function Base.getproperty(data::LazyStructuredData, name::Symbol)
    return if name in LAZY_STRUCTURED_INTERNALS
        getfield(data, name)
    elseif name in keys(data)
        data[name]
    else
        getfield(data, name)
    end
end

Base.size(data::LazyFieldArray) = field_shape(data.parent, data.fields)
Base.axes(data::LazyFieldArray) = map(Base.OneTo, size(data))
DiskArrays.haschunks(::LazyFieldArray) = DiskArrays.Unchunked()

function DiskArrays.readblock!(data::LazyFieldArray, out, ranges::OrdinalRange...)
    ndims(data) == length(ranges) || error("Number of indices is not correct")
    io = open_lazy_source(data.parent)
    try
        if !read_random_array_block!(io, data, out, ranges)
            for ci in CartesianIndices(out)
                index = ntuple(j -> ranges[j][ci[j]], ndims(data))
                out[ci] = field_cartesian_value(io, data, index)
            end
        end
    finally
        close(io)
    end
    return out
end

function read_random_array_block!(io::IO, data::LazyFieldArray, out, ranges)
    parent = data.parent
    parent.hdu_type == Random || return false
    length(data.fields) == 1 || return false
    field = first(data.fields)
    field.leng > 1 || return false
    is_unit_ranges(ranges) || return false

    inner_ranges = Base.tail(ranges)
    span = contiguous_linear_span(field.shape, inner_ranges)
    isnothing(span) && return false

    record_bytes = sizeof(parent.format.type) *
        (parent.format.param + prod(parent.format.shape))
    field_offset = first(field.slice) - 1 + sizeof(field.type) * (first(span) - 1)
    innershape = Base.tail(size(out))
    scale = get(parent.keywds, :scale, true)

    for (j, group) in pairs(first(ranges))
        seek(io, parent.begpos + record_bytes * (group - 1) + field_offset)
        values = read_bigendian(io, field.type, length(span))
        view(out, j, ntuple(_ -> Colon(), length(innershape))...) .=
            reshape(scaled_values(values, field, scale), innershape)
    end
    return true
end

function Base.read(data::LazyFieldArray)
    parent = data.parent
    if parent.hdu_type == Random && length(data.fields) == 1 &&
            first(data.fields).leng > 1
        return Array(data)
    end

    io = open_lazy_source(parent)
    return try
        if parent.hdu_type == Random && length(data.fields) > 1
            cols = [
                read(io, field, parent.format, parent.begpos; parent.keywds...)
                    for field in data.fields
            ]
            hcat(cols...)
        elseif length(data.fields) == 1
            read(io, first(data.fields), parent.format, parent.begpos; parent.keywds...)
        else
            Array(data)
        end
    finally
        close(io)
    end
end

Base.collect(data::LazyFieldArray) = read(data)

function field_linear_value(data::LazyFieldArray, index::Integer)
    shape = size(data)
    ci = Tuple(CartesianIndices(shape)[index])
    return field_cartesian_value(data, ci)
end

function field_cartesian_value(data::LazyFieldArray, ci::Tuple)
    io = open_lazy_source(data.parent)
    return try
        field_cartesian_value(io, data, ci)
    finally
        close(io)
    end
end

function field_cartesian_value(io::IO, data::LazyFieldArray, ci::Tuple)
    parent = data.parent
    return if parent.hdu_type == Random
        read_random_value(io, parent, data.fields, ci)
    elseif parent.hdu_type == Bintable
        read_bintable_value(io, parent, first(data.fields), ci)
    else
        read_table_value(io, parent, first(data.fields), first(ci))
    end
end

function read_random_value(data::LazyStructuredData, fields::Vector, ci::Tuple)
    io = open_lazy_source(data)
    return try
        read_random_value(io, data, fields, ci)
    finally
        close(io)
    end
end

function read_random_value(io::IO, data::LazyStructuredData, fields::Vector, ci::Tuple)
    group = first(ci)
    field = length(fields) == 1 ? first(fields) : fields[ci[2]]
    inner = length(fields) == 1 && length(ci) > 1 ?
        LinearIndices(field.shape)[Base.tail(ci)...] : 1
    offset = data.begpos +
        sizeof(data.format.type) * (data.format.param + prod(data.format.shape)) * (group - 1) +
        first(field.slice) - 1 + sizeof(field.type) * (inner - 1)
    seek(io, offset)
    value = read_bigendian(io, field.type)
    return scale_value(value, field, get(data.keywds, :scale, true))
end

function read_bintable_value(data::LazyStructuredData, field, ci::Tuple)
    io = open_lazy_source(data)
    return try
        read_bintable_value(io, data, field, ci)
    finally
        close(io)
    end
end

function read_bintable_value(io::IO, data::LazyStructuredData, field, ci::Tuple)
    row = first(ci)
    L, M = data.format.shape[1], first(field.slice) - 1
    seek(io, data.begpos + L * (row - 1) + M)
    return if !isnothing(field.pntr)
        K = read_bigendian(io, field.pntr)
        beg = read_bigendian(io, field.pntr)
        seek(io, data.begpos + data.format.heap + beg)
        values = [read_bigendian(io, field.type) for _ in 1:K]
        get(data.keywds, :scale, true) ? field.zero .+ field.scale .* values : values
    elseif field.type <: AbstractString
        rstrip(field.type(read(io, length(field.slice))))
    elseif field.type <: BitVector
        read(io, BitVector, field.leng)
    elseif field.leng == 1
        value = read_bigendian(io, field.type)
        scale_value(value, field, get(data.keywds, :scale, true))
    else
        inner = length(ci) > 1 ? ci[2] : 1
        seek(io, data.begpos + L * (row - 1) + M + sizeof(field.type) * (inner - 1))
        value = read_bigendian(io, field.type)
        scale_value(value, field, get(data.keywds, :scale, true))
    end
end

function read_table_value(data::LazyStructuredData, field, row::Integer)
    io = open_lazy_source(data)
    return try
        read_table_value(io, data, field, row)
    finally
        close(io)
    end
end

function read_table_value(io::IO, data::LazyStructuredData, field, row::Integer)
    L, M, leng = data.format.shape[1], first(field.slice) - 1, length(field.slice)
    seek(io, data.begpos + L * (row - 1) + M)
    item = String(read(io, leng))
    return if field.type <: AbstractString
        rstrip(item)
    elseif (!isnothing(field.null) && field.null == item) ||
            item == repeat(' ', leng) || leng <= 0
        missing
    else
        value = Base.parse(field.type, replace(item, "D" => "e", "E" => "e"))
        scale_value(value, field, get(data.keywds, :scale, true))
    end
end

function read_lazy_row(data::LazyStructuredData, row::Integer)
    (1 <= row <= structured_nrows(data)) || throw(BoundsError())
    pairs = Pair{Symbol, Any}[]
    for (name, ndx) in field_groups(data)
        values = [read_lazy_field_row_value(data, data.fields[j], row) for j in ndx]
        value = length(values) == 1 ? first(values) : hcat(values...)
        push!(pairs, name => value)
    end
    return (; pairs...)
end

function read_lazy_field_row_value(data::LazyStructuredData, field, row::Integer)
    array = LazyFieldArray(data, [field])
    return if data.hdu_type == Random && field.leng > 1
        out = Array{eltype(array)}(undef, field.shape)
        for ci in CartesianIndices(out)
            out[ci] = read_random_value(data, [field], (row, Tuple(ci)...))
        end
        out
    elseif data.hdu_type == Bintable && isnothing(field.pntr) &&
            !(field.type <: AbstractString) && !(field.type <: BitVector) && field.leng > 1
        [array[row, j] for j in 1:field.leng]
    else
        array[row]
    end
end
