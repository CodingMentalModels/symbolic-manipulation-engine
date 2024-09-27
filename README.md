# Symbolic Manipulation Engine

At the end of the day, math reduces to moving symbols around on a piece of paper. So why not let a computer do it instead?

The Symbolic Manipulation Engine is an extensible engine for defining allowable symbolic manipulations, saving them into "Contexts" so that they can be reused / shared / etc., and creating tooling for parsing input (e.g. from a command line) and displaying output (LaTeX, etc.), such that doing math, physics, etc. is faster / easier than doing it on paper / blackboard / whiteboard, etc.

It's currently under development, sometimes live at https://twitch.tv/codingmentalmodels.

A VSCode front-end is also under development but not currently public.

## Upcoming Features
- LaTeX support

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

## Type System

A type system is necessary because:
- It allows us to constrain the types of valid transformations in a super useful way.
- When we write math, we implicitly know what types things are, and notation, etc. reflects that.  

Our type system should:
- Model and enforce a `TypeHierarchy`, to allow e.g. $\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \ldots$.  
- Allow transformations to be predicated on the types of the variables involved.

Our `Symbols` then have:
- A name (how it appears)
- A return type
- Possibly children

Because `Type`s and `Symbols` aren't semantic,
- The engine is agnostic about what it means to have a statement.  Is it a true logical statement?  Is it one step in a series of calculations?  It doesn't matter.  It's just something that can be transformed with `Transformation`s.
- The engine doesn't care if you try to write $p \and \not p$ or set up the Barber's paradox.  It doesn't try to stop you, but what you can do with that is probably going to be uninteresting.

In order to facilitate higher order logic, there's a notion of an Arbitrary symbol in `Transformations`, which can be used to express things like:
- Leibnitz Rule: `p=q -> Any(p)=Any(q)`
- Mathematical Induction

## Deductive System

You can use the engine as a deductive system by supplying the various rules as `Transformation`s.  For example,
- $p, q$ -> $p \and q$

What's more, the type system allows you to get quantification for free.
- A `Type` can be viewed as a predicate, so a `Symbol` having that `Type` means that it satisfies the predicate.
- For all quantification is free when a `Symbol` has a `Type` corresponding to the Predicate you care about.
- Exists quantification can be done with a subtype corresponding to the particular element.

Universal Elimination: If we have a statement S(x:U) then we can always write S(c:U) for free for any c:U.
Universal Introduction: If we have a statement S(x:U) then we know that it holds for all U.
Existential Elimination: If we have a statement S(x:V) where V is a subtype of U, then we can always state that x:U.
Existential Introduction: If we have a statement S(x:V) where V is a subtype of U, then we know there's at least one x it holds for in U.

## A note on Godel's Incompleteness Theorem

TODO



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

### Context

A `Context` owns:
 - A `TypeHierarchy`, which it passes along to the Workspace (and validates) when it imports.
 - A list of allowed `Transformations`


### Workspaces

A `Workspace` is a scope for doing transformations, consisting of allowed transformations, statements (whether hypothesized or results of transformations), and their provenances.  

For example, suppose the user loads the `Context` for arithmetic, and then hypothesizes the statement $2 + 2$.  Then within the `Workspace` we can transform it using some arithmetic transformation to $2 + 2 + 1 - 1$ and the `Workspace` knows that this latter statement is derived from the former as well as a transformation ($x \implies x + y - y$).

Provenances should contain enough information to roll back statements, find all parents, find all children, etc.  



### Command Line Interface

In order to interact with the engine, we'll need an interface.  Let's start with a CLI, which allows:
- Initializing Workspaces
    - Under the hood, this creates a directory and a hidden directory, `.symbol`, which maintains state.  
    - `./.symbol/workspace.toml` contains a serialized version of the `Workspace`.  
- Defining new valid transformations
- Defining new contexts
- Importing contexts
- Stating statements
- Transforming statements

### Input Parsing

Inputting statements as trees is going to be painful, we want to be able to input them as plaintext and have them be parsed out.

Examples:
- `./sme hypothesize '2 + 2 = 4'` should add =(+(2, 2), 4) to the workspace and interpret the numbers as the broadest type that's in the current `Context`.  That means that the workspace needs to know not only the definitions of various symbols / transforms, but also their syntactic form, e.g.
    - `2` and `4` should be interpreted as `2 \in 2` and `4 \in 4` where those have certain (procedurally defined) properties.
    - `+` and `=` are `Infix`, binary operators, which should be interpreted as being over the broadest applicable type until otherwise specified.

This means that the parsing process should look like:
- Tokenize chunks of characters based on spaces, commas, parentheses, brackets, certain reserved characters like `+` and `!`.  
- For each token, check the active workspace for an `Interpretation`, which consists of criteria (e.g. `InputCriteria::Matches(SomeRegex)`) and a rule for how to interpret it as a type and the syntactic form (`Infix`, `Prefix`, etc.).  
- Given the interpretations, try to parse the output and kick out helpful errors where that's not possible.
- Allow parsing hints to disambiguate, e.g. it should always be possible to force a certain set of characters to be tokenized into one or force certain types by providing flags to the command.

### Pretty Printing

- `Symbol`s, `Statement`s, and `Transformations` can be given LaTeX representations, which can then be pretty printed.  

### Evaluation

It's going to be convenient to be able to evaluate arithmatic.  Create an extensible framework for allowing computation modules to run against certain `Type`s in certain `Context`s.  

### Further Ideas

- Embed this into a text editor, e.g. Sublime, VSCode, Neovim
- Automated Interpolation, e.g. we wouldn't want someone to have to spell out that:

$(15 + 3x + (5 + (2x^2 - 4)) + 7) \implies (15 + 5 - 4 + 7) + 3x + 2x^2$, it should be interpolated.
