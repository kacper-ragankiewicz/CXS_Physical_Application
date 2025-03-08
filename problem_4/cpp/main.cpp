#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <unordered_map>
#include <algorithm>

using namespace std;

// Function to load input data from a file
void load_input(const string &filename, int &L, int &T, double &p0, double &pk, double &dp)
{
    ifstream file(filename);
    if (!file)
    {
        cerr << "Failed to open file: " << filename << endl;
        exit(1);
    }

    string line;
    vector<string> data;
    while (getline(file, line))
    {
        size_t pos = line.find('%');
        if (pos != string::npos)
        {
            line = line.substr(0, pos);
        }
        data.push_back(line);
    }

    file.close();
    L = stoi(data[0]);
    T = stoi(data[1]);
    p0 = stod(data[2]);
    pk = stod(data[3]);
    dp = stod(data[4]);
}

// Function to generate the LxL lattice with occupancy probability p
vector<vector<int>> generate_lattice(int L, double p)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    vector<vector<int>> lattice(L, vector<int>(L, 0));
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            if (dis(gen) < p)
            {
                lattice[i][j] = 1;
            }
        }
    }
    return lattice;
}

// Function to perform the burning method for percolation
bool burning_method(const vector<vector<int>> &lattice)
{
    int L = lattice.size();
    vector<vector<int>> burned(L, vector<int>(L, 0));

    // Initialize the burning process
    vector<pair<int, int>> front;
    for (int i = 0; i < L; ++i)
    {
        if (lattice[0][i] == 1)
        {
            burned[0][i] = 1;
            front.push_back({0, i});
        }
    }

    while (!front.empty())
    {
        vector<pair<int, int>> new_front;
        for (auto &p : front)
        {
            int r = p.first, c = p.second;
            for (auto &dir : vector<pair<int, int>>{{-1, 0}, {1, 0}, {0, -1}, {0, 1}})
            {
                int nr = r + dir.first, nc = c + dir.second;
                if (nr >= 0 && nr < L && nc >= 0 && nc < L && lattice[nr][nc] == 1 && burned[nr][nc] == 0)
                {
                    burned[nr][nc] = 1;
                    new_front.push_back({nr, nc});
                }
            }
        }
        front = new_front;
    }

    // Check if the last row is reached
    for (int i = 0; i < L; ++i)
    {
        if (burned[L - 1][i] == 1)
        {
            return true;
        }
    }
    return false;
}

// Function to perform Hoshen-Kopelman algorithm for cluster labeling
unordered_map<int, int> hoshen_kopelman(const vector<vector<int>> &lattice)
{
    int L = lattice.size();
    vector<vector<int>> labels(L, vector<int>(L, 0));
    int label = 0;
    unordered_map<int, int> label_dict;

    for (int r = 0; r < L; ++r)
    {
        for (int c = 0; c < L; ++c)
        {
            if (lattice[r][c] == 0)
            {
                continue;
            }

            vector<int> neighbors;
            if (r > 0 && labels[r - 1][c] > 0)
            {
                neighbors.push_back(labels[r - 1][c]);
            }
            if (c > 0 && labels[r][c - 1] > 0)
            {
                neighbors.push_back(labels[r][c - 1]);
            }

            if (neighbors.empty())
            {
                label += 1;
                labels[r][c] = label;
                label_dict[label] = label;
            }
            else
            {
                int min_label = *min_element(neighbors.begin(), neighbors.end());
                labels[r][c] = min_label;
                for (int n : neighbors)
                {
                    label_dict[n] = min(label_dict[n], min_label);
                }
            }
        }
    }

    // Second pass to relabel clusters
    for (int r = 0; r < L; ++r)
    {
        for (int c = 0; c < L; ++c)
        {
            if (labels[r][c] > 0)
            {
                labels[r][c] = label_dict[labels[r][c]];
            }
        }
    }

    // Count cluster sizes
    unordered_map<int, int> cluster_sizes;
    for (int r = 0; r < L; ++r)
    {
        for (int c = 0; c < L; ++c)
        {
            if (labels[r][c] > 0)
            {
                cluster_sizes[labels[r][c]]++;
            }
        }
    }

    cluster_sizes.erase(0); // Remove the background cluster
    return cluster_sizes;
}

// Function to run the Monte Carlo simulation
void monte_carlo_simulation(int L, int T, double p0, double pk, double dp)
{
    vector<pair<double, pair<double, double>>> results;
    for (double p = p0; p <= pk; p += dp)
    {
        int flow_count = 0;
        vector<int> max_sizes;
        unordered_map<int, int> cluster_distributions;

        for (int i = 0; i < T; ++i)
        {
            // Generate lattice and run percolation test
            vector<vector<int>> lattice = generate_lattice(L, p);
            bool percolates = burning_method(lattice);
            flow_count += percolates;

            // Perform Hoshen-Kopelman to find clusters
            unordered_map<int, int> cluster_sizes = hoshen_kopelman(lattice);
            max_sizes.push_back(cluster_sizes.empty() ? 0 : max_element(cluster_sizes.begin(), cluster_sizes.end(), [](const pair<int, int> &a, const pair<int, int> &b)
                                                                        { return a.second < b.second; })
                                                                ->second);

            // Update cluster distribution
            for (const auto &cs : cluster_sizes)
            {
                cluster_distributions[cs.first] += cs.second;
            }
        }

        double Pf_low = static_cast<double>(flow_count) / T;
        double avg_smax = accumulate(max_sizes.begin(), max_sizes.end(), 0.0) / max_sizes.size();
        results.push_back({p, {Pf_low, avg_smax}});

        // Save the cluster distribution for this p
        string dist_filename = "Dist-p" + to_string(p) + "L" + to_string(L) + "T" + to_string(T) + ".txt";
        ofstream dist_file(dist_filename);
        for (const auto &dist : cluster_distributions)
        {
            if (dist.second > 0)
            {
                dist_file << dist.first << "  " << dist.second << endl;
            }
        }
    }

    // Save results for p, Pf_low, ⟨smax⟩
    string results_filename = "Ave-L" + to_string(L) + "T" + to_string(T) + ".txt";
    ofstream results_file(results_filename);
    for (const auto &result : results)
    {
        results_file << result.first << "  " << result.second.first << "  " << result.second.second << endl;
    }
}

int main()
{
    int L, T;
    double p0, pk, dp;

    // Load input data
    load_input("perc-ini.txt", L, T, p0, pk, dp);

    // Run the Monte Carlo simulation
    monte_carlo_simulation(L, T, p0, pk, dp);

    return 0;
}
