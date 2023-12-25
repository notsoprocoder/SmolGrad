# SmolGrad
### *because Karpathy and Hotz did it.*

*day 0*
Winter 2023, I haven't been coding much at work and I wanted to scratch the itch. I'd recently stumbled across TinyGrad and MicroGrad. It's been a long time since I've written an MLP, so why not do it from scratch.

## __RADD__
*day 1*
The first lesson I learned was simple dunder method, namely, `__radd__`. I wanted the `tensors.Vector` class to be summable with the in-built `sum` method. I'd implemented `__add__` and through that was sufficient.

Under the hood, it appears python's sum function looks something like...
``` python
def sum(array: ...) -> number:
    output = 0
    for i in array:
        output+=i
    return output
```
The challenge here is that we have `int(0)+Value(i)` and this would run in the translated addition of `Value(i) + int(0)` but to implement associative addition on the class the `__radd__` dunder needs to be applied.

## Elegance from Structure
*day 1-2*
Elegance is often seen as the implementation of a simple and efficient algorithm. There is also elegance in structure. But Elegance seems to be an ephemeral quality that is hard to define...

By structuring code in a consistant and 'logical' fashion, we can infer elegance. In *SmolGrad*, I've tried to add an elegant structure into the classes themselves. Now, it would be imodest to suggest it is elegant, but it is more elegant than the initial implementation. 

The class structure follows:
- definitions block: naturally, we need to initialise the class first. But we can group any methods within that define the class at the top, namely, `__repl__` in this case.
- Iter block: if the object is iterable, grouping these methods next. This is more an application of consistancy than any logical reasoning.
- Operations block: now we have our operators, I've ordered these by how primative they are, we start with add, then multiply, and finish with equality (as it naturally completes an equation).

We can extend this further by ensuring the same logic applies to our test suit and test the methods in this order. And similiarly, we can continue with the our logical ordering by ordering for example our type hints by how primative they are, for example, `int`->`float`->`Element`->`Vector`->`Matrix`. This can be extended to our if else blocks too. 

Ultimately, by taking some time to define order and applying rigourously we can create a consistant code base that 'feels good' and someone somewhere might define as elegant.



