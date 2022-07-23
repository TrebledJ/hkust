% ['./ass3.pl'].

/* The database of flight facts. */
flight(0, 1).
flight(3, 4).
flight(1, 2).
flight(2, 0).
flight(2, 1).
flight(3, 2).
flight(4, 3).

% Q1.

% BFS implementation.
bfsi(_V, [T | _], T). % Found target node.
bfsi(V, [C | Q], T) :- member(C, V), bfsi(V, Q, T), !.
bfsi(V, [C | Q], T) :-
    % Find all destinations connected to Current node.
    findall(D, (flight(C, D), \+ member(D, V), \+ member(D, Q)), NewNodes),
    append(Q, NewNodes, NextQ),
    bfsi([C | V], NextQ, T).
    
bfs(S, T) :- bfsi([], [S], T).

% Gets all flight nodes (source or destination).
all_nodes(F) :- findall(S, flight(S, _), Ss), findall(D, flight(_, D), Ds), append(Ss, Ds, R), list_to_set(R, F).

% Implementation of reachable.
reachablei([], _, _) :- false.
reachablei([S], AllSrcs, Dst) :- bfs(S, Dst), \+ member(Dst, AllSrcs), !. % Cut is to prevent next line being executed with the same [S | []] list.
reachablei([S | Srcs], AllSrcs, Dst) :-
    bfs(S, Dst),
    \+ member(Dst, AllSrcs), % Ensure Dst and AllSrcs are different.
    reachablei(Srcs, AllSrcs, Dst).

% Check if all given source nodes can reach `Dst`.
reachable(Srcs, Dst) :-
    all_nodes(All),
    member(Dst, All), % Check/generate `Dst`.
    (var(Srcs) -> % if `Srcs` hasn't been instantiated...
        % Generate `Srcs`.
        subset2(Srcs, All),
        reachablei(Srcs, Srcs, Dst)
    ; % Have a separate implementation that works with unsorted `Srcs`.
        reachablei(Srcs, Srcs, Dst)
    ).
    

% Tests/Generates subsets.

% Check S is a subset of L.
subset2(S, L) :- subset2i(S, [], L).
subset2i(S, A, []) :- S = A.
subset2i(S, A, [X | L]) :- subset2i(S, A, L); subset2i(S, [X | A], L).

% Q1 Test Suite.
?- \+ reachable([], 1).
?- reachable([1], 2).
?- reachable([3], 4).
?- reachable([4], 3).
?- setof(X, reachable([2,3], X), L), L = [0, 1].
?- setof(X, reachable([3,2], X), L), L = [0, 1].
?- setof(X, reachable([0], X), L), L = [1, 2].
?- findall(X, reachable([5], X), L), L = [].
?- findall(X, reachable(X, 5), L), L = [].
?- setof(X, reachable(X, 0), L),
    L = [[1],[1,3],[2],[2,1],[2,1,3],[2,3],[3],[4],[4,1],[4,1,3],[4,2],[4,2,1],[4,2,1,3],[4,2,3],[4,3]].


% Q2.

% Make list of nodes, in the order specified by the facts.
all_nodes2(F) :- findall([S, D], flight(S, D), Ps), reverse(Ps, R), foldl(append, R, [], F).
nubi([], Y, R) :- reverse(Y, R).
nubi([X | Xs], Y, R) :- member(X, Y), nubi(Xs, Y, R), !.
nubi([X | Xs], Y, R) :- nubi(Xs, [X | Y], R).
nub(X, Y) :- nubi(X, [], Y).

all_cities(L, N) :- all_nodes2(X), nub(X, Y), prefix(L, Y), length(L, N).

% Q2 Test Suite.
?- all_cities(L, 0), L = [].
?- all_cities(L, 2), L = [0, 1].
?- all_cities(L, 3), L = [0, 1, 3].
?- all_cities(L, 5), L = [0, 1, 3, 4, 2].
?- all_cities([0, 1, 3], 3).
?- \+ all_cities([0, 1, 3], 5).
?- \+ all_cities([4, 1, 0], 3).


% Q3.

% Helper for cons and append.
cons_append(Xs, X, X0, Xout) :- append([[X | Xs]], X0, Xout).

% BFS implementation, keeps tracks of paths, exhausts all nodes.
% Paths are reversed for easy access to last node.
bfsi2([], _T, Done, R) :- R = Done. % Explored all paths.
bfsi2([Pth | Pths], T, Done, R) :- % Path has reached target. Save first.
    Pth = [T | _],
    bfsi2(Pths, T, [Pth | Done], R),
    !. % Don't run next implementation.
    
bfsi2([Pth | Pths], T, Done, R) :-
    Pth = [Last | _], % Unpack current path.
    % Get nodes following the last node.
    findall(D, (flight(Last, D), \+ member(D, Pth)), Nexts),
    % Append following nodes to current path.
    foldl(cons_append(Pth), Nexts, [], NewPths),
    % Queue the new path.
    append(Pths, NewPths, NextPths),
    bfsi2(NextPths, T, Done, R)
    .
    
% Get all paths starting from S.
bfs2(S, T, Pths) :- bfsi2([[S]], T, [], Pths).

count_paths(Src, Dst, N) :-
    % Generate all Src, Dst, and N first...
    all_nodes(All),
    length(All, Max),
    numlist(1, Max, Range),
    member(Src, All), member(Dst, All), member(N, Range),
    Src \== Dst,
    % Then filter Src, Dst based on BFS and satisfaction... 
    bfs2(Src, Dst, Pths),
    length(Pths, N).

% Q3 Test Suite.
?- count_paths(0, 1, N), N = 1.
?- count_paths(0, 2, N), N = 1.
?- count_paths(2, 1, N), N = 2.
?- findall([X, Y], count_paths(X, Y, 2), L), L = [[3, 1], [2, 1], [4, 1]].
?- findall([X, N], count_paths(X, X, N), L), L = [].
?- findall([X, N], count_paths(X, 2, N), L), L = [[0, 1], [3, 1], [1, 1], [4, 1]].
?- findall([X, N], count_paths(2, X, N), L), L = [[0, 1], [1, 2]]. 
?- setof([X, Y, N], count_paths(X, Y, N), L), L = [[0,1,1],[0,2,1],[1,0,1],[1,2,1],[2,0,1],[2,1,2],[3,0,1],[3,1,2],[3,2,1],[3,4,1],[4,0,1],[4,1,2],[4,2,1],[4,3,1]].


% Q4.

length2(N, X) :- length(X, N).

shortest_paths(Src, Dst, L) :-
    % Generate all Src, Dst, and N first...
    all_nodes(All),
    member(Src, All), member(Dst, All),
    Src \== Dst,
    % Then filter Src, Dst based on BFS and satisfaction... 
    bfs2(Src, Dst, Pths),
    maplist(reverse, Pths, Pths2),
    maplist(length, Pths, Lens),
    min_member(N, Lens),
    include(length2(N), Pths2, L).

% Q4 Test Suite.
?- shortest_paths(0, 1, L), L = [[0, 1]].
?- shortest_paths(4, 2, L), L = [[4, 3, 2]].
?- shortest_paths(2, 1, L), L = [[2, 1]].
?- shortest_paths(0, 1, [[0, 1]]).
?- \+ shortest_paths(2, 1, [[2, 0, 1]]).
?- setof([X, Y, Z], shortest_paths(X, Y, Z), L),
    L = [[0,1,[[0,1]]],[0,2,[[0,1,2]]],[1,0,[[1,2,0]]],[1,2,[[1,2]]],[2,0,[[2,0]]],[2,1,[[2,1]]],[3,0,[[3,2,0]]],[3,1,[[3,2,1]]],[3,2,[[3,2]]],[3,4,[[3,4]]],[4,0,[[4,3,2,0]]],[4,1,[[4,3,2,1]]],[4,2,[[4,3,2]]],[4,3,[[4,3]]]].


% Q5.

% BFS implementation, stops after a certain distance.
% The queue stores pairs of city ID and distance from start.
bfsi3(V, [], _N, R) :- R = V.
bfsi3(V, [C-N | Q], N, R) :- bfsi3([C | V], Q, N, R), !.
bfsi3(V, [C-M | Q], N, R) :-
    P is M+1,
    % Find all destinations connected to Current node.
    findall(D-P, (flight(C, D), \+ member(D, V), \+ member(D-_, Q)), NewNodes),
    append(Q, NewNodes, NextQ),
    bfsi3([C | V], NextQ, N, R).
    
bfs3(S, N, Connected) :- bfsi3([], [S-0], N, Connected).


search_destination(L, Len, Src) :-
    all_nodes(All),
    length(All, Max),
    RealMax is Max - 1,
    numlist(0, RealMax, Range),
    member(Src, All),
    member(Len, Range),
    bfs3(Src, Len, Connected),
    (nonvar(L) ->
        sort(L, L2),
        sort(Connected, C2),
        delete(C2, Src, L2)
    ; delete(Connected, Src, L)
    ).


% Q5 Test Suite.
?- search_destination(L, 0, 0), L = [].
?- search_destination(L, 1, 0), L = [1].
?- search_destination(L, 2, 0), sort(L, L2), L2 = [1, 2].
?- search_destination([1], 1, 0).
?- \+ search_destination([0, 1], 2, 0).
?- findall(Src, search_destination([1], 1, Src), L), L = [0].
?- findall(Src, search_destination([0,2], 2, Src), L), L = [1].
?- findall(Src, search_destination([2,0], 2, Src), L), L = [1].
?- findall(Src, search_destination([2,4,1,0], 4, Src), L), L = [3].
?- findall([Len,Src], search_destination([1,2], Len, Src), L), L = [[2,0],[3,0],[4,0]].
?- findall([L, Len], search_destination(L, Len, 2), L), L = [[[],0],[[1,0],1],[[1,0],2],[[1,0],3],[[1,0],4]].
