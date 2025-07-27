### 2.1 Understanding Unicode

(a) It's the NULL character

(b) The string representation of chr(0) is '\x00', while the printed
representation is not visible

(c) The print statement renders text and shows "this is a teststring", while the
string representation prints invisible character "this is a test\x00string"

### 2.2 Unicode Encodings

(a) 1. utf-8 is dominant for web texts 2. utf-8 encoded bytes are more
space-efficient, where it uses 1 byte for ascii chars, comparing with utf-16 (2
bytes) and utf-32 (4 bytes). 3. UTF-8 naturally aligns with byte-level
processing, while UTF-16 and UTF-32 introduce complexity during grouping common
bytes.

(b) Example: café will be decoded as 'cafÃ©'. This function decodes the input
utf-8 strings one byte at a time, while the correct way is to decode the whole
string together.

(c) 0xC0 0xAF is a 2-byte sequence that does not decode to any Unicode character
in UTF-8 because it violates the encoding rules (overlong form)

### 2.5 BPE Training on TinyStories

(a) Training took around 8 minutes, used 5GB RAM. The longest token is
"accomplishment".

(b) Pretokenization takes most of the time & resource.
