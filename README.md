# Record
*Track your time, right in your terminal. It just works.*

Record helps you master your most valuable asset: your time. Effortlessly track the hours you dedicate to every project. And when it’s time to bill or review, generate beautiful, insightful reports right from your command line.

## Installation

As of now, you can use `pipx` to install Record in your machine:

```sh
pipx install git+https://github.com/avantguarda/record
 ```

## Setting Up the Development Environment
To set up the development environment for Record, you'll need to have Python 3.13 installed. You can use [uv](https://docs.astral.sh/uv/) to manage dependencies and run commands.

1. **Clone the Repository:**

   ```sh
   git clone <repository-url>
   cd record
   ```

2. **Install uv:**

   If you don't have uv installed, you can install it by following the instructions on the [uv installation page](https://docs.astral.sh/uv/#installation).

3. **Install Dependencies:**

   ```sh
   uv sync
   ```

   This command will install all the necessary dependencies for both the application and development.

4. **Activate the Virtual Environment:**

   ```sh
   source .venv/bin/activate
   ```

## Running the Application
To run the application, use the following command:

```sh
rec
```

## Acknowledgements
Every great tool has a story. Ours begins with `watson` time tracker, created by [Jazzband](https://jazzband.co/). It was a powerful, elegant solution that many of us came to rely on.

After nearly three years of inactivity on the [original project](https://github.com/jazzband/Watson/), we wanted to ensure its legacy would not only survive but thrive. Record was created to carry the torch forward.

As a direct fork, `rec` builds upon the brilliant foundation of `watson`. Our goal is to provide the active maintenance, modern enhancements, and long-term support this project deserves. We are deeply grateful to the original contributors for building something worth preserving.

## License
Record is released under the GNU General Public License 3.0, see [LICENSE](LICENSE) file for details.
