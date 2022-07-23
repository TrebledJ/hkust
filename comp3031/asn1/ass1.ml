(* redacted *)

(* First, let's open this can of worms. *)

open List;
structure O = Option;
structure A = Array;

(* And define a bunch of helper functions for composition. *)
val sort = ListMergeSort.sort;
fun fst (x, _) = x; val src = fst;
fun snd (_, y) = y; val dst = snd;

(* Curriable operators. *)
fun eq x y = x = y;
fun ne x y = x <> y;
fun gt x y = x > y;
fun lt x y = x < y;
fun le x y = x <= y;
fun ge x y = x >= y;

type city = int;

(* Next, let's create a couple types with obscure constructors. *)
datatype flight = F of city * city;
datatype flights = Fs of flight list;

fun unF (F f) = f;

(* I like safe code. *)
fun assert (name: string, cond: bool) = if cond then print (name^": ok\n") else raise (Fail name);


(* And we're ready to begin! *)

(*            *)
(* Question 1 *)
(*            *)

fun reachable (Fs fs: flights, (from, to): city * city): bool = 
    let val out_edges = filter (fn F(f, _) => f = from) fs
        fun try [] = false
          | try (F(f, t)::xs) = let val rest_edges = Fs(filter (fn F(f', _) => f' <> f) fs)
                                in  reachable (rest_edges, (t, to)) orelse try xs   (* Recurse and pass on remaining edges. *)
                                end                                                 (* This prevents loops. *)
    in  from = to orelse try out_edges (* Base case: from == to. Recurse on remaining flights otherwise. *)
    end; (* Remember thy semi-colons. *)


(*            *)
(* Question 2 *)
(*            *)

(* Returns a list of tuples where the first field is the count of the matches of the second field.
 * A match between two elements is determined by the predicate `f`.
 *)
fun groupby (f: 'a * 'a -> bool) (xs: 'a list): (int * 'a) list =
    let fun go (x, []) = [(1, x)]
          | go (x, ((c, y)::acc)) = if f (x, y) then (c+1, y)::acc else (1, x)::(c, y)::acc
    in  foldr go [] xs
    end;
    
fun popular_cities (Fs fs: flights): city list =
    let val data = (sort (fn (x, y) => fst x < fst y)  (* Sort so that largest count is in front. *)
                    o groupby op=            (* Count and aggregate equal numbers. *)
                    o sort op<              (* Sort so that same numbers are next to each other. *)
                    o map (dst o unF))      (* Get list of destinations. *)
                    fs
    in  (map snd o filter (fn (c, x) => c = fst (hd data))) data
    end;


(*            *)
(* Question 3 *)
(*            *)

val sum = foldr op+ 0

fun count_paths (Fs fs: flights, (from, to): city * city): int =
    let val fs' = map unF fs
        val fs_from = filter (eq from o src) fs' (* Out-edges (destinations) from `from`. *)
        val next_fs = (filter (ne from o dst) o filter (ne from o src)) fs' (* Remove any connections to `from`. *)
        val subpaths = map (fn flt => count_paths (Fs(map F next_fs), (dst flt, to))) fs_from
    in  if from = to then 1 else if null fs then 0 else sum subpaths
    end;


(*            *)
(* Question 4 *)
(*            *)

type path = city list;

(* Remove duplicates from a list. *)
fun nub xs =
    let fun nubgo [] ys = rev ys
          | nubgo (x::xs) ys = if exists (eq x) ys then nubgo xs ys else nubgo xs (x::ys)
    in nubgo xs []
    end;

(* Each recursive call to shortest_paths_go will advance one level down the BFS tree.
 * This ensures that if a certain shortest path is found, then *all* shortest paths are 
 * found in the same iteration.
 *
 * Paths are stored in reverse so that we can easily access the latest node from the lists' head.
 *
 * The check function should return an option:
 *  - NONE if bfs should continue,
 *  - SOME (x) if bfs should terminate and return x.
 *)
fun bfs (fs: (city * city) list, check: (path list -> path list option), ps: path list): path list =
    case check ps of
        O.SOME ret => ret
      | O.NONE => 
            let 
                fun go (p: path, acc: path list) = 
                    let
                        fun visited c = exists (eq c) p (* Checks if a city has been visited in this path. *)

                        (* Only check direct flights from `hd p` and new destinations. *)
                        val fs' = filter (fn (s, d) => s = hd p andalso not (visited d)) fs
                    in
                        (* If no flights, then continue. *)
                        if null fs' then
                            acc
                        else (* Otherwise prepend the new flights to `p` and add all *those* flights to acc. *)
                            map (fn flt => dst flt::p) fs' @ acc
                    end
            in
                bfs (fs, check, foldr go [] ps)
            end;

fun shortest_paths_check (to: city) (ps: path list): path list option =
    if null ps then
        O.SOME []                           (* No paths... sad. :( *)
    else if exists (eq to o hd) ps then
        O.SOME (filter (eq to o hd) ps)    (* At least one path has reached the target. Return said path(s). *)
    else
        O.NONE;

fun shortest_paths (Fs fs: flights, (from, to): city * city): path list = 
    (map rev o bfs) (map unF fs, (shortest_paths_check to), [[from]]);


(*            *)
(* Question 5 *)
(*            *)

type 'a matrix = 'a A.array A.array;

fun matrix (m: int) (n: int) (x: 'a): 'a array array =
    A.tabulate (n, fn _ => A.array (m, x));

fun rows (M: 'a matrix): int = A.length M;
fun cols (M: 'a matrix): int = if rows M = 0 then 0 else A.length (A.sub (M, 0));

infix matsub;
infix #=;
infix #*;
infix #^;
infix notin;
infix index;

(* Access an element of the matrix. *)
fun op matsub (M: 'a matrix, (row: int, col: int)): 'a =
    A.sub (A.sub (M, row), col);

(* Assigns a value to an element in a matrix. *)
fun op #= (M: 'a matrix, (i: int, j: int, x: 'a)): unit =
    A.update (A.sub (M, i), j, x);

fun op notin (a: ''a, xs: ''a list): bool =
    case xs of
        [] => true
      | (x::xs') => x <> a andalso a notin xs';

fun op index (array: ''a A.array, x: ''a): int =
    case A.findi (fn (_, x') => x = x') array of
        O.NONE => ~1
      | O.SOME (i, _) => i;

fun loop (n: int) (f: (int -> unit)): unit = (A.tabulate (n, f); ());

fun floyd_warshall (M: int matrix, n: int): unit =
    let fun go (k, i, j) =
            let val a = M matsub (i, k)
                val b = M matsub (k, j)
            in
                if a + b < (M matsub (i, j)) then
                    M #= (i, j, a + b) (* Update min distance. *)
                else
                    ()
            end;
    in
        loop n (fn k => loop n (fn i => loop n (fn j => go (k, i, j))))  (* Modifies M directly. *)
    end;

fun search_cities (Fs fs: flights, L: int): (city * city) list =
    let val fs' = map unF fs
        val srcs = map fst fs'
    
        (* An array containing a mapping of indices to city id. *)
        val mapping = A.fromList (nub srcs @ (filter (fn x => x notin srcs) o nub o map snd) fs')

        (* Encode flights as an adjacency matrix. *)
        val msize = A.length mapping
        val mfs =
            let val init = matrix msize msize 9999
            in
                (* Set matrix elements to true for each (i, j) in flights. *)
                (app (fn (i, j) => init #= (mapping index i, mapping index j, 1)) fs';
                init)
            end

        (* Helper function to collect reachable pairs. *)
        fun go i (j, dist, acc) = 
            if dist <= L then 
                (A.sub (mapping, i), A.sub (mapping, j))::acc  (* Map to city id and add to accumulator. *)
            else
                acc (* i -> j can't be reached. Don't modify accumulator. *)
        
        fun matrix_to_pairs m =
            A.foldri (fn (i, row, acc) => (* First level. *)
                A.foldri (go i) [] row @ acc) (* Flatten the list by concatenating directly with outer accumulator. *)
                    [] m
    in
        (floyd_warshall (mfs, msize); matrix_to_pairs mfs)
    end;


(*            *)
(* Test Cases *)
(*            *)

(
assert ("reachable test 1", reachable (Fs [], (0, 1)) = false);
assert ("reachable test 2", reachable (Fs [F(0,1), F(1,0)], (0, 1)) = true);
assert ("reachable test 3", reachable (Fs [F(0,1), F(1,2), F(2,0)], (0, 2)) = true);
assert ("reachable test 4", reachable (Fs [F(0,1), F(1,0), F(1,2)], (0, 2)) = true);
assert ("reachable test 5", reachable (Fs [F(0,2), F(2,1)], (1, 0)) = false);
assert ("popular_cities test 1", popular_cities (Fs []) = []);
assert ("popular_cities test 2", popular_cities (Fs [F(0,1), F(1,0)]) = [1,0]);
assert ("popular_cities test 3", popular_cities (Fs [F(0,1), F(0,2), F(1,0), F(2,1)]) = [1]);
assert ("popular_cities test 4", popular_cities (Fs [F(0,1), F(2,1), F(3,1), F(4,1), F(0,2), F(1,0), F(5,1), F(1,2), F(3,2), F(4,2), F(5,2)]) = [2,1]);
assert ("popular_cities test 5", popular_cities (Fs [F(0,1), F(2,1), F(3,1), F(4,1), F(0,2), F(1,0), F(5,1), F(1,2), F(3,2), F(4,2), F(5,2), F(6,2)]) = [2]);
assert ("count_paths test 1", count_paths (Fs [], (0, 1)) = 0);
assert ("count_paths test 2", count_paths (Fs [F(0,1), F(1,0)], (0, 1)) = 1);
assert ("count_paths test 3", count_paths (Fs [F(0,1), F(0,2), F(1,2), F(1,3), F(2,0), F(3,2)], (0, 2)) = 3);
assert ("count_paths test 4", count_paths (Fs [F(0,1), F(0,2), F(1,3), F(2,3), F(3,4), F(3,5), F(4,6), F(5,6)], (0, 6)) = 4);
assert ("count_paths test 5", count_paths (Fs [F(0,2), F(2,0), F(1,3), F(3,1)], (0, 1)) = 0);
assert ("shortest_paths test 1", shortest_paths (Fs [], (0, 1)) = []);
assert ("shortest_paths test 2", shortest_paths (Fs [F(0,1), F(1,0)], (0, 1)) = [[0,1]]);
assert ("shortest_paths test 3", shortest_paths (Fs [F(0,1), F(0,3), F(1,2),F(1,3), F(2,0), F(3,2)], (0, 2)) = [[0,1,2],[0,3,2]]);
assert ("shortest_paths test 4", shortest_paths (Fs [F(0,1), F(0,2), F(1,3), F(2,3), F(3,4), F(3,5), F(4,6), F(5,6)], (0, 6)) = [[0,1,3,4,6],[0,1,3,5,6],[0,2,3,4,6],[0,2,3,5,6]]);
assert ("shortest_paths test 5", shortest_paths (Fs [F(2,1), F(1,2)], (0, 2)) = []);
assert ("search_cities test 1", search_cities (Fs [], 1) = []);
assert ("search_cities test 2", search_cities (Fs [F(0,1), F(1,0)], 1) = [(0,1),(1,0)]);
assert ("search_cities test 3", search_cities (Fs [F(0,2), F(1,0), F(2,1)], 2) = [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]);
assert ("search_cities test 4", search_cities (Fs [F(0,1), F(0,2), F(1,3), F(2,3)], 3) = [(0,1),(0,2),(0,3),(1,3),(2,3)]);
assert ("search_cities test 5", search_cities (Fs [F(0,1), F(1,0)], 2) = [(0,0),(0,1),(1,0),(1,1)]);
assert ("search_cities test 6", search_cities (Fs [F(0,1), F(1,0)], 3) = [(0,0),(0,1),(1,0),(1,1)]);
assert ("search_cities test 7", search_cities (Fs [F(0,1), F(1, 10), F(2,0), F(2,11)], 3) = [(0,1),(0,10),(1,10),(2,0),(2,1),(2,10),(2,11)]);
print "all tests passed\n");
