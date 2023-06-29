# Autogenerated using ProtoBuf.jl v1.0.11 on 2023-06-19T18:18:24.286
# original file: /home/lior/TensorBoardLogger.jl/gen/proto/tensorboard/compat/proto/types.proto (proto3 syntax)

import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export var"#DataType", SerializedDType

@enumx var"#DataType" DT_INVALID=0 DT_FLOAT=1 DT_DOUBLE=2 DT_INT32=3 DT_UINT8=4 DT_INT16=5 DT_INT8=6 DT_STRING=7 DT_COMPLEX64=8 DT_INT64=9 DT_BOOL=10 DT_QINT8=11 DT_QUINT8=12 DT_QINT32=13 DT_BFLOAT16=14 DT_QINT16=15 DT_QUINT16=16 DT_UINT16=17 DT_COMPLEX128=18 DT_HALF=19 DT_RESOURCE=20 DT_VARIANT=21 DT_UINT32=22 DT_UINT64=23 DT_FLOAT8_E5M2=24 DT_FLOAT8_E4M3FN=25 DT_FLOAT_REF=101 DT_DOUBLE_REF=102 DT_INT32_REF=103 DT_UINT8_REF=104 DT_INT16_REF=105 DT_INT8_REF=106 DT_STRING_REF=107 DT_COMPLEX64_REF=108 DT_INT64_REF=109 DT_BOOL_REF=110 DT_QINT8_REF=111 DT_QUINT8_REF=112 DT_QINT32_REF=113 DT_BFLOAT16_REF=114 DT_QINT16_REF=115 DT_QUINT16_REF=116 DT_UINT16_REF=117 DT_COMPLEX128_REF=118 DT_HALF_REF=119 DT_RESOURCE_REF=120 DT_VARIANT_REF=121 DT_UINT32_REF=122 DT_UINT64_REF=123 DT_FLOAT8_E5M2_REF=124 DT_FLOAT8_E4M3FN_REF=125

struct SerializedDType
    datatype::var"#DataType".T
end
PB.default_values(::Type{SerializedDType}) = (;datatype = var"#DataType".DT_INVALID)
PB.field_numbers(::Type{SerializedDType}) = (;datatype = 1)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:SerializedDType})
    datatype = var"#DataType".DT_INVALID
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            datatype = PB.decode(d, var"#DataType".T)
        else
            PB.skip(d, wire_type)
        end
    end
    return SerializedDType(datatype)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::SerializedDType)
    initpos = position(e.io)
    x.datatype != var"#DataType".DT_INVALID && PB.encode(e, 1, x.datatype)
    return position(e.io) - initpos
end
function PB._encoded_size(x::SerializedDType)
    encoded_size = 0
    x.datatype != var"#DataType".DT_INVALID && (encoded_size += PB._encoded_size(x.datatype, 1))
    return encoded_size
end