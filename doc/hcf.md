# Heterogeneous container format (HCF) documentation

In all [compilation flows](compilation.md) in which AdaptiveCpp takes responsibility for embedding device code in the host binary, the embedded device code will be in the HCF format.

HCF is a format that has been designed to be easy to parse, and allow binary content as well as text-based metadata to coexist in a flexible hierarchical structure.

The AdaptiveCpp runtime can be instructed to dump the HCF data embedded in the application by setting the environment variable `HIPSYCL_HCF_DUMP_DIRECTORY` [(details)](env_variables.md).

`acpp-hcf-tool` can be used to inspect or alter HCF files.

## HCF definition

```
<HCF> ::= <ReadableHeader>'__acpp_hcf_binary_appendix'<BinaryAppendix>
<ReadableHeader> ::= <NodeContentLines>
<NodeContentLines> ::= <KeyValueLine> | <Subnode> | <KeyValueLine><NodeContentLines> | <Subnode><NodeContentLines> | ø
<KeyValueLine> ::= [<Whitespaces>]<Key>[<Whitespaces>] '=' [<Whitespaces>]<Value>[<Whitespaces>]'\n'
<Subnode> ::= [<Whitespaces>] '{.'<UniqueSubnodeName> [<Whitespaces>] '\n' 
    <NodeContentLines> 
    [<Whitespaces>] '}.' <UniqueSubnodeName> [<Whitespaces>]  '\n'
```

`UniqueSubnodeName` is a string that is unique among all subnodes of the parent node.

`<BinaryAppendix>` is a binary blob that contains the concatenated binary data from all nodes. Binary data is attached to nodes by adding a subnode:
```
{.__binary
  start=<StartOffsetInBinaryAppendix>
  size=<SizeInBytes>
}.__binary
```

## Example

The following HCF data contains a root node, two subnodes, and one binary appendix containing 'ABC':

```
a = 123
b = asd64
{.MySubnode
  keyname = Hello
  {.__binary
    start = 0
    size = 3
  }.__binary
  key2 = World
}.MySubnode
{.MySubnode2
  keyname = Turtle
}.MySubnode2
__acpp_hcf_binary_appendixABC
```