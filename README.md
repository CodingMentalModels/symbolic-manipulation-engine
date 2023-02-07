# Symbolic Manipulation Engine

Goal: An extensible engine for defining allowable symbolic manipulations, saving them into "Contexts" so that they can be reused / shared / etc., and creating tooling for parsing input (e.g. from a command line) and displaying output (LaTeX, etc.), such that doing math, physics, etc. is faster / easier than doing it on paper / blackboard / whiteboard, etc.

## User Stories

### Arithmetic

Given an arithmetic expression, like: $\ln(15) + 3 \cdot 2^2 - \sin(\tau/3)$, allow simplifications and computations of the result.  e.g.

$$
\begin{align}

\ln(15) + 3 \cdot 2^2 - \sin(\tau/3) = \\
\ln(15) + 12 - \sqrt{3}/2 = \\
13.84202...
\end{align}
$$


### Algebra

Given an algebraic expression like \$x^3 - 3x^2 + 3x - 1\$, factor and simplify it.

Manipulations with complex numbers, manipulations with Eisenstein Integers, manipulations with Clifford Algebras, etc.  Because each of these can be defined via a series of formal transformations on symbols, this should all be possible.

Example `Transformation`s:
- $x \in \mathbb{Z}, y \in \mathbb{Z}, + \in +: \enspace x + y \implies y + x$

### Proofs

Given several hypothesized statements, manipulate them to reach a new conclusion.  E.g. given: $ x \in 2\mathbb{Z}, \enspace y \in 2\mathbb{Z} $, we should be able to show that $x + y \in 2\mathbb{Z}$.  

Something like:
$$
\begin{align}
\exists z \in \mathbb{Z} \enspace z = x + y \\
\exists a \in \mathbb{Z} \enspace 2a = x \\
\exists b \in \mathbb{Z} \enspace 2b = y \\
z = 2a + 2b \\
z = 2 \cdot (a + b) \\
\exists c \in \mathbb{Z} \enspace c = a + b \\
z \in 2\mathbb{Z}
\end{align}
$$

Many other similar cases, including:
- Set theory
- Propositional logic
- Calculus
- Physics

In each case, the point is to have a `Context` (or several) which provide valid transformations of symbols in that context.  


## Features

### Core Manipulations

- `Symbol`s are pure syntax; they have no inherent meaning but have `Type`s, format strings, and (optionally) `LaTeX` representations.
- `Type`s are also purely syntactic; they restrict the types of transformations that are valid.  For example:
    - Integer (arity = 0), e.g. 2, 5, -2, x
    - 2 (arity = 0), uniquely corresponds to the number 2.  Has $2 \in \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \ldots$ etc. as part of its type hierarchy.
    - Proposition (arity = 0), e.g. p, q
    - `+` (arity = 2, variable types, but we can define it on e.g. Quaternions and it'll apply to everything else in the type hierarchy)
    - Function (arity = 2, the domain and the range)

- `Statement`s are trees of `Symbol`s
- `Transformation`s are rules for mapping one or more `Statement`s to a new `Statement`.  
- `Context`s are groups of saved, allowable `Transformation`s which can be imported and used, e.g. one might have a `Basic Algebra` `Context`, which can perform algebraic manipulations.

### Command Line Interface

In order to interact with the engine, we'll need an interface.  Let's start with a CLI, which allows:
- Defining new valid transformations
- Defining new contexts
- Importing contexts
- Stating statements
- Transforming statements

### Input Parsing

Inputting statements as trees is going to be painful, we want to be able to input them as plaintext and have them be parsed out.

### Database

- Stores and retreives `Context`s
- Stores and retreives saved work

### Pretty Printing

- `Symbol`s, `Statement`s, and `Transformations` can be given LaTeX representations, which can then be pretty printed.  

### Evaluation

It's going to be convenient to be able to evaluate arithmatic.  Create an extensible framework for allowing computation modules to run against certain `Type`s in certain `Context`s.  

### Further Ideas

- Embed this into a text editor, e.g. Sublime, VSCode, Neovim
- Automated Interpolation, e.g. we wouldn't want someone to have to spell out that:

$(15 + 3x + (5 + (2x^2 - 4)) + 7) \implies (15 + 5 - 4 + 7) + 3x + 2x^2$, it should be interpolated.

