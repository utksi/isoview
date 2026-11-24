# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Utkarsh Singh
"""
    Isoview

Show files in isoview (https://utkarsh.science/isoview/) inside a notebook (Pluto, IJulia, VS Code).

`isoview(...)` returns a `View` whose `text/html` form is an iframe of the isoview site in embedded mode (`?embed`)
plus a short script. When the embedded page reports `{isoview: "ready"}` by postMessage, the script sends it the files
as `{isoview: "open", files: [{name, base64}], mode: "replace"}`. Nothing is uploaded: the iframe and the files meet
in the user's browser. Files are sent uncompressed (base64 is 4/3 of the bytes); gzip-compressed files (`.gz`) are
sent as they are and unpacked by isoview.
"""
module Isoview

using Base64

export isoview

"The address of the isoview site."
const SITE = "https://utkarsh.science/isoview/"
"Largest total of file bytes per view; larger files would bloat the notebook."
const LIMIT = 200 * 2^20

"""
    View(files, height, query, site)

An isoview viewer for a notebook cell: `files` is a vector of `name => bytes`, `height` in CSS pixels, `query` the
extra URL parameters of the embedded page (theme, style, look, panel), `site` the isoview address.
"""
struct View
    files::Vector{Pair{String,Vector{UInt8}}}
    height::Int
    query::Vector{Pair{String,String}}
    site::String
end

"`name => bytes` for one argument: a file path, or `name => content` with content bytes or a string."
_item(path::AbstractString) = basename(path) => read(path)
_item(p::Pair{<:AbstractString,<:AbstractVector{UInt8}}) = String(p.first) => Vector{UInt8}(p.second)
_item(p::Pair{<:AbstractString,<:AbstractString}) = String(p.first) => Vector{UInt8}(codeunits(p.second))
_item(x) = throw(ArgumentError("isoview cannot use a $(typeof(x)): give a file path or name => bytes"))

"""
    isoview(items...; height=480, theme=nothing, style=nothing, look=nothing, panel=false, site=SITE)

Show the given files in one isoview scene in the notebook cell. `items`: file paths (any format isoview opens:
POSCAR, CIF, CHGCAR, LOCPOT, cube, WAVECAR with its POSCAR, trajectories, ...) or `name => bytes` pairs.
`theme`: "light" or "dark" (default: follows the browser). `style`: "ball", "stick", "space". `look`: "classic",
"soft", "glossy", "toon", "outline". `panel`: show the side panel at start.
"""
function isoview(items...; height::Integer=480, theme=nothing, style=nothing, look=nothing, panel::Bool=false, site::AbstractString=SITE)
    isempty(items) && throw(ArgumentError("isoview needs at least one file"))
    files = Pair{String,Vector{UInt8}}[_item(x) for x in items]
    total = sum(length(last(f)) for f in files)
    total > LIMIT && throw(ArgumentError("$(round(Int, total / 2^20)) MB is too much to keep in a notebook (limit $(LIMIT ÷ 2^20) MB); open these files in $site or the desktop app"))
    query = Pair{String,String}[k => string(v) for (k, v) in (("theme", theme), ("style", style), ("look", look)) if v !== nothing]
    panel && push!(query, "panel" => "1")
    View(files, Int(height), query, String(site))
end

"JSON string literal of `s` (quotes, backslashes, control characters and `</` escaped for use inside a script)."
function _js(s::AbstractString)
    io = IOBuffer()
    print(io, '"')
    for c in s
        if c == '"' || c == '\\'
            print(io, '\\', c)
        elseif c == '<'
            print(io, "\\u003c")
        elseif c < ' '
            print(io, "\\u", string(UInt32(c), base=16, pad=4))
        else
            print(io, c)
        end
    end
    print(io, '"')
    String(take!(io))
end

"Characters of `s` escaped for an HTML attribute value."
_attr(s::AbstractString) = replace(replace(replace(s, "&" => "&amp;"), "\"" => "&quot;"), "<" => "&lt;")

function Base.show(io::IO, ::MIME"text/html", v::View)
    q = join(["embed"; ["$k=$(v)" for (k, v) in v.query]], "&")
    src = v.site * "?" * q
    origin = join(split(v.site, "/")[1:3], "/")
    fid = "isoview-" * string(rand(UInt64), base=16)
    files = join(["{\"name\":" * _js(name) * ",\"base64\":\"" * base64encode(data) * "\"}" for (name, data) in v.files], ",")
    print(io, "<iframe id=\"", fid, "\" src=\"", _attr(src), "\" style=\"width:100%;height:", v.height,
          "px;border:0;border-radius:10px\" allowfullscreen title=\"isoview\"></iframe>",
          "<script>(function(){var f=document.getElementById(\"", fid, "\"),files=[", files, "];",
          "window.addEventListener('message',function(e){if(f&&e.source===f.contentWindow&&e.data&&",
          "e.data.isoview==='ready')f.contentWindow.postMessage({isoview:'open',files:files,mode:'replace'},",
          _js(origin), ");});})();</script>")
end

Base.show(io::IO, v::View) = print(io, "Isoview.View(", join(first.(v.files), ", "), ")")

end # module
